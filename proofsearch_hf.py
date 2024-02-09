from accelerate import Accelerator
from pylean import LeanServer
import torch
import heapq
import concurrent
import transformers
import os
import logging
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    StoppingCriteria,
    StoppingCriteriaList,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


# -----------------------------------------------
# PREP

accelerator = Accelerator()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
logger = logging.getLogger()


class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
# -----------------------------------------------


# -----------------------------------------------
# ENV
def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def _load_data(dataset_name: str='minif2f-test',
               dataset_path: str='./data/minif2f.jsonl'):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data
# -----------------------------------------------


# -----------------------------------------------
# PROMPT

def _prompt_fewshot(tactic_state):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

Tactic state:
----
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
----
Next tactic:
----
rintro s t ⟨u, a, hr, he⟩
----

Tactic state:
----
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
----
Next tactic:
----
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
----

Tactic state:
----
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
----
Next tactic:
----
rw [← h.gcd_eq_one]
----

In your response, include only the lean code for only the next tactic and nothing else.
Tactic state:
----
%s
----
Next tactic:
----""" % (tactic_state)
    return prompt
# -----------------------------------------------


# -----------------------------------------------
# LOADING MODELS

def load_model_vllm(model_name: str='open-web-math/llemma_7b',
                    tp_degree: int=1,
                    dtype: str='float16',
                    max_num_batched_tokens: int=4096):

    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_model_hf(model_name):
    if 'pythia' in model_name:
        model = transformers.GPTNeoXForCausalLM.from_pretrained(
            model_name,
            device_map='auto')
        tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name)
    else:
        # Set the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Set the model
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    low_cpu_mem_usage=True)

        # Set the eos
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare the model
    model = accelerator.prepare(model)

    # Set the model in eval mode (to edit later)
    model.eval()
    return model, tokenizer
# -----------------------------------------------


# -----------------------------------------------
# GENERATION

## Helper
def _unique_sorted(texts, scores):
    """
    Sort the texts according to log prob.
    """
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

## For VLLM
def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):

    # Init textx and scores
    texts, scores = [], []

    for temperature in temperatures:
        # Set the params
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
        )

        # Get the outputs
        outputs = model.generate([prompt], params, use_tqdm=False)
        if len(outputs) == 0:
            return [], []

        # Get texts and scores
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


## For HF
def sequence_scores(out, prompt_length, model, tokenizer, stop_div='----'):
    """
    Returns each output sequence's log probability normalized by the number of tokens.
    An output sequence is defined as the tokens after the prompt up to and including eos.
    """

    # Get the text
    # TODO: The text should be trimmed
    text = tokenizer.batch_decode(out.sequences)

    # Trim the text
    # Unlike VLLM, the stop words will be retained, so we have to manually remove
    for i, text_i in enumerate(text):
        if f'\n{stop_div}</s>' in text_i:
            text[i] = text_i.replace(f'\n{stop_div}</s>', '</s>')

        if f'\n{stop_div[:-1]}</s>' in text_i:
            text[i] = text_i.replace(f'\n{stop_div[:-1]}</s>', '</s>')

    input_ids = tokenizer(
        text, return_tensors="pt", padding='longest', truncation=True
    ).to(model.device)

    with torch.no_grad():
        # Get the probs
        out = model(**input_ids)
        probs = torch.log_softmax(out.logits, dim=-1).detach()
        probs = probs[:, :-1, :]

        # Get the probs after the prompt
        input_ids_shifted = input_ids.input_ids[:, 1:]
        log_probs = torch.gather(probs, 2, input_ids_shifted[:, :, None]).squeeze(-1)
        log_probs = log_probs[:, prompt_length:]
        up_to_eos_mask = (input_ids_shifted[:,prompt_length:].eq(
            tokenizer.eos_token_id).cumsum(1).cumsum(1) <= 1).type(log_probs.dtype)

        # Normalize the scores
        normalized_sequence_scores = (log_probs * up_to_eos_mask).sum(1) / up_to_eos_mask.sum(1)

    return normalized_sequence_scores


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to('cuda') for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        # print(tokenizer.decode(last_token), last_token, len(input_ids[0]))
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False


def trim_output(output_text, stop_div='---'):
    """
    Trims the generated output text to remove the stop sequence and everything after it.

    Parameters:
    - output_text (str): The generated text.
    - stop_sequence (str): The sequence after which the text should be trimmed.

    Returns:
    - str: The trimmed text.
    """
    stop_index = output_text.find(stop_div)
    if stop_index != -1:
        # Trim the output to just before the stop sequence
        return output_text[:stop_index]
    else:
        # If the stop sequence is not found, return the original text
        return output_text


def generate_hf(prompt, model, tokenizer, temperatures, num_samples, stopping_criteria):

    # Init texts and scores
    texts, scores = [], []

    # Get the input ids
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Generate
    with torch.no_grad():
        # Does beam search at temp 0.0, otherwise temperature sampling.
        for temp in temperatures:
            decoding_params = dict(
                max_new_tokens=256,
                do_sample=temp > 0,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria,
            )
            if temp == 0.0:
                decoding_params['num_beams'] = num_samples

            # Get the output
            out = model.generate(
                input_ids, **decoding_params
            )

            # Get the texts
            # TODO: Apply the text trimmining
            decoded_seqs = tokenizer.batch_decode(
                out.sequences[:,input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Remove that '---'
            decoded_seqs = [trim_output(text,
                                 stop_div='---').strip()
                            for text in decoded_seqs]

            # Extend to the texts
            texts.extend(decoded_seqs)

            # Get the scores
            scores_ = sequence_scores(
                out=out,
                prompt_length=input_ids.shape[1],
                model=model,
                tokenizer=tokenizer
            )
            scores.extend(scores_.view(-1).tolist())

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores
# -----------------------------------------------


# -----------------------------------------------
# SEARCH
def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        prompt_fn,
        timeout=600,
        early_stop=False,
        max_tokens=256,
        stopping_criteria,
) -> dict:
    """
    Best first search.
    """
    # Initialize the results
    attempt_results = []

    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            # ------------------------------------------------
            # PREPARATION
            start = time.time()
            proof_finished = False
            queue = [(0.0, [], init_state, [])]
            visited = set()
            # ------------------------------------------------

            # ------------------------------------------------
            # STEP BY STEP INFERENCE
            for iteration in trange(max_iters):

                # ---------------------------------------------
                # Preparation
                # Termination criteria
                if len(queue) == 0 or proof_finished: break

                # Get the information from the heapq
                total_score, steps, state, trace = heapq.heappop(queue)
                ts = _tactic_state(state)
                logger.info(f'\nCurrent State:\n{ts}\n')
                visited.add(ts)

                # ---------------------------------------------
                # Generate results
                assert args.gen_method in ['vllm', 'hf']
                if args.gen_method == 'vllm':
                    step_cands, step_scores = generate_vllm(
                        prompt_fn(ts),
                        model,
                        tokenizer,
                        temperatures,
                        num_samples,
                        stop='----',
                        max_tokens=max_tokens
                    )

                elif args.gen_method == 'hf':
                    step_cands, step_scores = generate_hf(
                        prompt=prompt_fn(ts),
                        model=model,
                        tokenizer=tokenizer,
                        temperatures=[0.0],
                        num_samples=args.num_samples,
                        stopping_criteria=stopping_criteria)

                step_cands = [s.strip() for s in step_cands]
                # ---------------------------------------------

                # ---------------------------------------------
                # Update the queue
                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        'tactic': step,
                        'state_before': _tactic_state(state)
                    }

                    # When the proof is finished
                    if isinstance(result, ProofFinished):
                        attempt_results.append({
                            'theorem': theorem.full_name,
                            'proof': steps + [step],
                            'score': total_score - score,
                            'success': True,
                            'failure_reason': '',
                            'trace': trace + [step_trace],
                            'temperature': temperatures,
                            'elapsed': start - time.time(),
                            'iteration': iteration
                        })
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        logger.info('Proof is finished for this theorem.')
                        break

                    # When there is still unsolved goals
                    elif isinstance(result, TacticState):
                        if _tactic_state(result) not in visited:
                            # Score is negative log probability summed across steps
                            new_score = (total_score - score)
                            heapq.heappush(
                                queue,
                                (new_score, steps + [step], result, trace + [step_trace])
                            )
                            logger.info(f'\nstep: {step}; score: {round(score, 3)}')
                # ---------------------------------------------

    except (DojoInitError, DojoHardTimeoutError, DojoCrashError, subprocess.CalledProcessError) as e:
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })
        logger.info('Crashed.')

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'SearchEnded'
        })
        logger.info('Search ended with no success.')

    return attempt_results
# -----------------------------------------------
