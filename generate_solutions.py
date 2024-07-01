import ast
import asyncio
from human_eval.data import write_jsonl, read_problems
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing_extensions import Annotated
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
import instructor
from instructor.exceptions import InstructorRetryException
import os

api_key=os.environ["OPENAI_API_KEY"]
# following https://python.useinstructor.com/blog/2023/11/13/learn-async/#example-batch-processing
client = instructor.apatch(AsyncOpenAI(api_key=api_key), mode=instructor.Mode.TOOLS)


class CodeSolution(BaseModel):
    explanation: str = Field(..., description="Explanation of how the solution works.")
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

async def generate_one_completion(prompt, semaphore) -> CodeSolution:
    async with semaphore:
        try:
            solution = await client.chat.completions.create(
                # model="gpt-4o",
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
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
    completions = await asyncio.gather(*[task for task in tasks if asyncio.iscoroutine(task)])
    samples = []
    for task_id in problems_sample:
        print(f"Generating solutions for task {task_id}...")
        for completion in completions:
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

