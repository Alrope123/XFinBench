import logging
import re
import signal
from importlib.metadata import version
from typing import Dict, List, Optional
from utils_hendrycks_math import (
    is_equiv as hendrycks_is_equiv,
    process_results as hendrycks_process_results,
)
import datasets


eval_logger = logging.getLogger(__name__)


try:
    import sympy
    from sympy.parsing.latex import parse_latex

    # assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    return normalize_final_answer(x1) == normalize_final_answer(x2)



def normalize_final_answer(answer_string: str) -> str:
    if answer_string != "":
        s = answer_string.strip().lower()

        if re.fullmatch(r"(true|t|yes|y|1)", s):
            answer_string = "true"
        elif re.fullmatch(r"(false|f|no|n|0)", s):
            answer_string = "false"
    return answer_string

def extract_answers(results):
    # Flexible answer extraction for multiple answers, the first one is the "primary" answer
    raw_answer = (
        results[0] if isinstance(results, list) else results
    )  # Only first result is used if list
    # boxed_answer = last_boxed_only_string(raw_answer)
    # if boxed_answer is not None:
    #     try:
    #         boxed_answer = remove_boxed(boxed_answer)
    #     except AssertionError:
    #         boxed_answer = None
    answer_format_regex = r"(?:therefore|thus|so|hence)?[, ]*(?:my )?(?:final )?(?:answer|result)\s*(?:is|=|:)?\s*((?:true|false|True|False|TRUE|FALSE))"
    prefix_regexes = [
        r"(?:therefore|thus|so|hence)[, ]*(?:my )?(?:final )?(?:answer|result)\s*(?:is|=|:)?",
        r"(?:the )?(?:correct )?(?:answer|result)\s*(?:is|=|:)?",
        r"(?:my )?answer\s*(?:is|=|:)?",
    ]
    answer_regexes = [
        r"\b(?:true|false|True|False|TRUE|FALSE)\b",
    ]
    all_answers = []
    num_answer = extract_answer_text(
        raw_answer,
        answer_format_regex=answer_format_regex,
        prefix_regexes=prefix_regexes,
        answer_regexes=answer_regexes,
    )["answer"]
    if num_answer is not None:
        all_answers.append(normalize_final_answer(num_answer))
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(raw_answer))
    return all_answers

def process_results(ground_answer, results):
    metrics: dict = {}

    max_flex_match = metrics.get("exact_match", 0)
    all_extracted_answers = extract_answers(results)
    for answer in all_extracted_answers:
        if max_flex_match == 1:
            break
        if is_equiv(answer, ground_answer):
            max_flex_match = 1
    metrics["exact_match_flex"] = max_flex_match
    if all_extracted_answers:
        metrics["model_answer"] = all_extracted_answers[0]
    return metrics


def extract_answer_text(
    continuation: str,
    task_config=None,
    answer_format_regex=None,
    answer_regexes_templates=None,
    answer_regexes=None,
    prefix_regexes=None,
    use_last_prefix_match=True,
    use_last_raw_match=True,
):
    """
    Input: model continuation, task config (contain the answer regexes - exact, prefix, just answer),
            if any of the prefix match whether to use last one, whether to use last of raw answer match
    Output: extracted answer, answer format correctness score (both in a dict)
    """

    if task_config is not None:
        answer_format_regex = answer_format_regex or task_config["metric_kwargs"].get(
            "answer_format_regex"
        )
        prefix_regexes = prefix_regexes or task_config["metric_kwargs"].get("answer_prefix_regexes")
        answer_regexes = answer_regexes or task_config["metric_kwargs"].get("answer_regexes")
        answer_regexes_templates = answer_regexes_templates or task_config["metric_kwargs"].get(
            "answer_regexes_templates"
        )

    answer_format_correct = 0.0
    answer_string = ""
    ## Try exact answer format regex first
    if answer_format_regex:
        matches = re.findall(answer_format_regex, continuation)
        if matches:
            # Pick the last occurrence, answer format correct score stays 1.0
            answer_string = matches[-1]
            answer_format_correct = 1.0

    ### Search for exact matches using the regex template
    if (answer_string == "") and answer_regexes_templates:
        for idx, template in enumerate(answer_regexes_templates):
            for answer_regex in answer_regexes:
                if "$ANS$" in template:
                    regex = template.replace("$ANS$", answer_regex)
                else:
                    regex = template

                matches_list = list(re.finditer(regex, continuation))
                if matches_list:
                    match = matches_list[-1] if use_last_prefix_match else matches_list[0]
                    groups = match.groups()
                    if groups:
                        answer_string = next((g for g in reversed(groups) if g), groups[0])
                    else:
                        answer_string = match.group(0)
                    if answer_string != "":
                        answer_format_correct = 1.0 if idx == 0 else 0.5
                        break
            if answer_string != "":
                break

    ## Search continuation for any prefix regex, extract first answer following it using answer regexes
    if (answer_string == "") and prefix_regexes:
        for idx, prefix_regex in enumerate(prefix_regexes):
            matches_list = list(re.finditer(prefix_regex, continuation))
            if len(matches_list) > 0:
                if use_last_prefix_match:
                    match = matches_list[-1]
                else:
                    match = matches_list[0]

                answer_text = continuation[match.end() :].strip().strip(".")
                # search for answer at the start of the rest:
                answer_match = re.findall("(" + "|".join(answer_regexes) + ")", answer_text)
                if answer_match:
                    res_tuple = answer_match[0]  # pick first one that follows prefix
                    if not isinstance(res_tuple, str):
                        answer_string = res_tuple[0]
                        for res1 in res_tuple[1:]:
                            if res1 != "":
                                answer_string = res1
                    else:
                        answer_string = res_tuple
                    if answer_string != "":
                        if idx == 0:
                            answer_format_correct = 1.0
                        else:
                            answer_format_correct = 0.5
                        break
    if answer_string == "":
        for idx, answer_regex in enumerate(answer_regexes):
            ans_match = re.findall(answer_regex, continuation)
            if ans_match:
                if use_last_raw_match:
                    answer_string = ans_match[-1]
                else:
                    answer_string = ans_match[0]
                if idx == 0:
                    answer_format_correct = 0.2
                else:
                    answer_format_correct = 0.1
                break
    return {"answer": answer_string, "answer_format_correct": answer_format_correct}

