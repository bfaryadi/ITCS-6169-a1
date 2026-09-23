# AI Usage

## Tools used

- GitHub Copilot (as an AI pair programmer)
- OpenAI / ChatGPT-style iterative prompting (if applicable in your workflow)

## Representative ways AI assisted

1. It helped me convert the starter notebook logic into a cleaner Python training script.
2. It suggested ways to structure the training loop, validation logic, and checkpoint selection.
3. It helped me reason about the likely source of poor performance when the baseline model was too weak.
4. It assisted in creating a more organized repository layout and documentation.
5. It helped check for obvious errors in the training/evaluation pipeline before I ran large experiments.

## An ineffective or questionable suggestion

One suggestion was to set `num_workers=2` and add a custom DirectML workaround without checking the actual model training behavior. This did not necessarily improve the model itself and introduced extra complexity without clear gains.

I verified the suggestion by comparing the training pipeline behavior and checking runtime stability. The training loop was then simplified to the more reliable version that matched the assignment’s baseline expectations and avoided unnecessary threading overhead.

## Important decision I made myself

I decided to keep the main classifier a CNN trained from scratch instead of jumping to a heavier architecture or external foundation-model approach. This keeps the task aligned with the assignment rules, makes the training pipeline interpretable, and allows us to run controlled experiments on data preprocessing, regularization, and architecture choices.

## Verification habit

Whenever I used AI-generated code, I checked:

- whether the tensor shapes matched the model architecture;
- whether the validation loop was actually measuring the correct metric;
- whether the checkpoint logic reflected the best validation accuracy; and
- whether the change was justified by an experiment, not just by a vague claim that it would be faster or more accurate.
