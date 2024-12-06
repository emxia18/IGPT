import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import textwrap
import os
import json

gpt_corr, gpt_inc = 13+8+5+4+8+6+13+8+7, 5+10+13+14+10+12+5+10+11
bot_corr, bot_inc = 11+13+7+14+16+15+8+9+14, 7+5+11+4+2+3+10+8+4
og_corr, og_inc = 7+12+16+16+16+16, 10+6+2+2+2+2

print(gpt_corr, gpt_inc, bot_corr, bot_inc, og_corr, og_inc)
print("gpt acc: ", (gpt_corr / (gpt_corr + gpt_inc)))
print("bot acc: ", (bot_corr / (bot_corr + bot_inc)))
print("emily acc: ", (og_corr / (og_corr + og_inc)))

categories = ("GPT", "Fine Tuned LLM", "Emily")
correct = [gpt_corr, bot_corr, og_corr]
incorrect = [gpt_inc, bot_inc, og_inc]

x = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(5, 5
))

bar1 = ax.bar(x, correct, label='Correct', color='lightgreen', edgecolor='black')
bar2 = ax.bar(x, incorrect, bottom=correct, label='Incorrect', color='lightcoral', edgecolor='black')

ax.set_xlabel("Categories")
ax.set_ylabel("Question Count")
ax.set_title("Human Evaluator Accuracy Distribution")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(title="Accuracy")

plt.tight_layout()
plt.savefig("IGPT/src/evaluation_methods/category_accuracy.png")

accuracies = [22,19,16,16,16,16,15,15,14,14,14,14,13,13,12,12,11,10]

plt.figure(figsize=(8, 6))
plt.hist(accuracies, bins=range(min(accuracies), max(accuracies) + 2), align='left', edgecolor='black', color='lightcoral')

plt.title('Human Accuracy (from 24)')
plt.xlabel('Accuracies')
plt.ylabel('Count')

plt.savefig("IGPT/src/evaluation_methods/accuracy.png")