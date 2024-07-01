import ast
import asyncio
from human_eval.data import write_jsonl, read_problems
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing_extensions import Annotated
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
import instructor
from instructor.exceptions import InstructorRetryException, IncompleteOutputException
import os

api_key=os.environ["OPENAI_API_KEY"]
# following https://python.useinstructor.com/blog/2023/11/13/learn-async/#example-batch-processing
client = instructor.apatch(AsyncOpenAI(api_key=api_key), mode=instructor.Mode.TOOLS)


class CodeSolution(BaseModel):
    explanation: str = Field(..., description="Explanation of how the solution works.")
    additional_tests: str = Field(..., description="additional test cases - inputs and expected outputs")
    draft_code: str = Field(..., description="A first pass at the problem")
    reflection: str = Field(..., description="Review of the draft code")
    code: str = Field(..., description="The code solution to the problem. A runnable python script.")

    @field_validator("code")
    @classmethod
    def validate_code(cls, code: str) -> bool:
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            print("SyntaxError")
            raise e

problems = read_problems()

semaphore = asyncio.Semaphore(10)

filler = """
# Programming and Problem-Solving Guidance

- Take a step back and reframe the problem if you're stuck.
- Consider multiple approaches before settling on one.
- Consider additional test cases to better understand the problem
- Consider whether your solution would work on the test cases

Here's an example problem:

Problem:
from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    \"\"\"Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    \"\"\"

Explanation:
The `filter_by_substring` function takes a list of strings and a substring as input. It filters the list to only include strings that contain the given substring. The function iterates through each string in the input list and checks if the substring is present in the string. If the substring is found, the string is added to the result list. Finally, the function returns the filtered list of strings.
    
More example test cases:
filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'ba') == ['bacd']
filter_by_substring(['lorem', 'ipsum', 'dolor'], 'o') == ['lorem', 'dolor']
filter_by_substring(['1234', '1255710', '81056710'], '80') == []

Draft Code:
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    filtered_strings = [s for s in strings if substring in s]
    return filtered_strings

Reflection:
In this case the draft code is adequate for the task. Reviewing it, the test cases should all pass
filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'ba') == ['bacd']
filter_by_substring(['lorem', 'ipsum', 'dolor'], 'o') == ['lorem', 'dolor']
filter_by_substring(['1234', '1255710', '81056710'], '80') == []

Proposed Solution:
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    filtered_strings = [s for s in strings if substring in s]
    return filtered_strings

Now, let's restate the problem I originally asked.

"""

async def generate_one_completion(prompt, semaphore) -> CodeSolution:
    async with semaphore:
        try:
            solution = await client.chat.completions.create(
                model="gpt-4o",
                # model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt + filler + prompt}],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.0,
                response_model=CodeSolution,
                max_retries=AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=16),
                ),
            )
            print(f"Generated solution for task")
            return solution
        except InstructorRetryException as e:
            print(e)
            return CodeSolution(
                explanation="Failed to generate valid code", 
                additional_tests="", 
                draft_code="", 
                reflection="", 
                code=""
            )
        except IncompleteOutputException as e:
            print(e)
            return CodeSolution(explanation="Failed to generate valid code", code="")

async def main():
    num_samples_per_task = 1
    # problem_count = 10
    problem_count = 200  # 164
    problems_sample = list(problems.keys())[:problem_count]
    tasks = []
    samples = []
    for task_id in problems_sample:
        print(f"Generating async task for task {task_id}...")
        for _ in range(num_samples_per_task):
            tasks.append(generate_one_completion(problems[task_id]["prompt"], semaphore=semaphore))
    print(f"# of Tasks: {len(tasks)}")
    completions = await asyncio.gather(*[task for task in tasks if asyncio.iscoroutine(task)])
    samples = []
    for task_id, completion in zip(problems_sample, completions):
        print(f"Generating solutions for task {task_id}...")
        samples.append(
            dict(
                task_id=task_id,
                completion=completion.code,
                explanation=completion.explanation,
                problem=problems[task_id]
            )
        )
    write_jsonl("samples.jsonl", samples)

if __name__ == "__main__":
    asyncio.run(main())

