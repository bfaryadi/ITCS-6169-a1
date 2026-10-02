1. Coding tools used: Gemini, Claude, CoPilot

2. Representative ways AI helped me:
    - I. CoPilot helped me set up DirectML so I could train on my AMD GPU. It directly edited portions of my code, although I avoided this in the future.
    - II. Asked Gemini, and later Claude, for various data augmentations I could try out. Some of them I kept, while some of them (like random crop and random rotation) seemed to hinder accuracy.
    - III. Gemini suggested reducing the batch size when I prompted it regarding the wild fluctuating of train and val loss. This changed yielded one of the largest accuracy increases in all the experiments, and the reasoning makes sense to me.
    - IV. I prompted Claude and Gemini both with information about the dataset size to see what model structure they would recommend (specifically regarding how many convolutional layers). Claude suggested a setup involving two conv-batch-relu layers per stage, with a maxpool after each. I was not aware of this as an option, and thought I had exhausted my options outside of simply adding more layers or changing the widths. This turned out to be very effective.
    - V. Claude and Gemini generated the MixUp and CutMix implementations. Claude also warned that combining the two mixes may create data too noisy, so I made it so only one can apply at once.
    - VI. Claude helped me split my notebook into 3 separate Python files after finishing experiments. I also had it implement saving model weights to the disk at the same time, which I really should have done sooner.
    - VII. CoPilot generated README.md from the repository information.

3. Incorrect/ineffective/questionable AI suggestion: Early on, Gemini suggested adding an additional linear layer as a method to avoid overfitting, but I hadn't given it very many details about my dataset or model. Later on, I found out that the linear layer wasn't making a difference, and was possibly hindering generalization.

4. How I tested/verified/corrected/rejected that suggestion: As with a few other suggestions, I re-tested the linear layer later on and concluded based on the loss/accuracy graphs that it wasn't helpful and was unnecessarily increasing model training time.

5. Important architectural decisions I made: Many of the data augmentations, including MixUp and CutMix, were ideas I wanted to implement from class, and not because of suggestions from AI. That being said, my overall model design was heavily influenced by AI. However, I believe I have done my due diligence in considering whether or not the changes make sense and testing them in isolation. I would have liked to rely a little less on AI to come up with ideas, but given my unfamiliarity with the field, I believe this has been a necessary learning experience.
