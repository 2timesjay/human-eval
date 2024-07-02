import ast
import asyncio
import random
from human_eval.data import write_jsonl, read_problems
from human_eval.execution import check_correctness
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing_extensions import Annotated
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
import instructor
from instructor.exceptions import InstructorRetryException
import os

api_key=os.environ["OPENAI_API_KEY"]
# following https://python.useinstructor.com/blog/2023/11/13/learn-async/#example-batch-processing
client = instructor.apatch(AsyncOpenAI(api_key=api_key))


class CodeSolution(BaseModel):
    extra_tests: str = Field(
        ..., 
        description="Additional test inputs and expected outputs, of the form def check(candidate): assert candidate(<input>) == <output> ..."
    )
    explanation: str = Field(..., description="Explanation of how the solution works.")
    code: str = Field(..., description="The code solution to the problem. A runnable python script.")
    solution_name: str = Field(..., description="The name of the solution function.")

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

When approaching any programming task, it's crucial to break down the problem into smaller, manageable steps. Start by clearly understanding the requirements and desired outcome. Then, outline a high-level strategy before diving into the code.

Consider the following principles:

1. **Simplicity**: Aim for clear, readable code. Avoid over-engineering solutions when simpler approaches suffice.

2. **Modularity**: Break your code into logical functions or modules. This improves readability and makes your code easier to test and maintain.

3. **Efficiency**: While premature optimization is often discouraged, be mindful of the efficiency of your algorithms, especially for larger datasets.

4. **Error Handling**: Anticipate potential issues and handle exceptions gracefully. Your code should be robust against unexpected inputs.

5. **Testing**: Regularly test your code with various inputs, including edge cases. This helps catch bugs early and ensures your solution works as intended.

When faced with a challenging problem:

- Take a step back and reframe the problem if you're stuck.
- Consider multiple approaches before settling on one.
- Use pseudocode to outline your logic before writing actual code.
- Don't hesitate to break complex operations into smaller, helper functions.
- Leverage appropriate data structures and built-in functions to simplify your code.

Remember, there's often more than one correct way to solve a problem. Focus on writing clear, correct code first, then optimize if necessary. Regular practice with diverse problems will enhance your problem-solving skills and expand your programming toolkit.

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

Test function:
def check(candidate):
    assert candidate([], 'a') == []
    assert candidate(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']
    assert candidate(['aaaaaaazzz', 'pizza', 'kazam'], 'za') == ['pizza', 'kazam']
    assert candidate(['flux', 'flummox', 'flake'], 'fl') == ['flux', 'flummox', 'flake']

Explanation:
The `filter_by_substring` function takes a list of strings and a substring as input. It filters the list to only include strings that contain the given substring. The function iterates through each string in the input list and checks if the substring is present in the string. If the substring is found, the string is added to the result list. Finally, the function returns the filtered list of strings.

Proposed Solution:
from typing import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    filtered_strings = [s for s in strings if substring in s]
    return filtered_strings

Solution Name: # must be exact name of solution function, used for testing
filter_by_substring
    

Now, let's restate the problem I originally asked.

"""

async def generate_one_completion(prompt, semaphore) -> CodeSolution:
    async with semaphore:
        solutions = []
        solution = ""
        try:
            solution = await client.chat.completions.create(
                # model="gpt-4o",
                model="gpt-3.5-turbo",
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
            solutions.append(solution)
            print(f"Generated solution for task")
        except InstructorRetryException as e:
            print(e)
            solution = CodeSolution(
                extra_tests="", 
                explanation="Failed to generate valid code", 
                code="",
                solution_name="",
            )
        for i in range(2,4):
            extra_tests = solution.extra_tests
            try:
                ast.parse(extra_tests)
                pseudo_problem = {
                    "task_id": None,
                    "entry_point": solution.solution_name,
                    "prompt": prompt,
                    "test": extra_tests
                }
                try:
                    # dict(
                    #     task_id=problem["task_id"],
                    #     passed=result[0] == "passed",
                    #     result=result[0],
                    #     completion_id=completion_id,
                    # )
                    execution = check_correctness(pseudo_problem, solution.code, 3.0, None)
                    if execution["passed"]:
                        print("Passed self-generate tests")
                        return solution
                except Exception as e:
                    print(f"Exception while checking correctness: {e}")
                    return solution
            except SyntaxError as e:
                print(e)
            try:
                solution = await client.chat.completions.create(
                    # model="gpt-4o",
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt + filler + prompt + "Previous solution attempt: " + solution.code}],
                    max_tokens=2000,
                    n=1,
                    stop=None,
                    temperature=0.2,
                    response_model=CodeSolution,
                    max_retries=AsyncRetrying(
                        stop=stop_after_attempt(3),
                        wait=wait_exponential(multiplier=1, min=2, max=16),
                    ),
                )
                print(f"Regenerated solution for task, attempt {i}")
                solutions.append(solution)
            except InstructorRetryException as e:
                print(e)
        return solution
            

async def main():
    num_samples_per_task = 1
    problem_count = 10
    problems_sample = random.sample(list(problems.keys()), problem_count)
    # problems_sample = list(problems.keys())
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
                extra_tests=completion.extra_tests,
                explanation=completion.explanation,
                problem=problems[task_id]
            )
        )
    write_jsonl("samples.jsonl", samples)

if __name__ == "__main__":
    asyncio.run(main())

