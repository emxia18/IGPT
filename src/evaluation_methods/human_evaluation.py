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
correct = [gpt_corr / (gpt_corr + gpt_inc), bot_corr / (bot_corr + bot_inc), og_corr / (og_corr + og_inc)]
incorrect = [gpt_inc / (gpt_corr + gpt_inc), bot_inc / (bot_corr + bot_inc), og_inc / (og_corr + og_inc)]

nb_corr = [5 / 9, 7 / 9, 4 / 6]
nb_inc = [4 / 9, 2 / 9, 2 / 6]

x = np.arange(len(categories))
bar_width = 0.3
space = 0.05

fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.bar(x - (bar_width + space) / 2, correct, bar_width, label='Human Correct', color='lightgreen', edgecolor='black')
bar2 = ax.bar(x - (bar_width + space) / 2, incorrect, bar_width, bottom=correct, label='Human Incorrect', color='lightcoral', edgecolor='black')

bar3 = ax.bar(x + (bar_width + space) / 2, nb_corr, bar_width, label='Naive Bayes Correct', color='darkgreen', edgecolor='black')
bar4 = ax.bar(x + (bar_width + space) / 2, nb_inc, bar_width, bottom=nb_corr, label='Naive Bayes Incorrect', color='darkred', edgecolor='black')

ax.set_xlabel("Categories")
ax.set_ylabel("Question Percentage")
ax.set_title("Human vs Naive Bayes Evaluator Accuracy Distribution")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(title="Accuracy")

plt.tight_layout()
plt.savefig("IGPT/src/evaluation_methods/graphs/emily_category_accuracy.png")

accuracies = [22,19,16,16,16,16,15,15,14,14,14,14,13,13,12,12,11,10]

plt.figure(figsize=(8, 6))
plt.hist(accuracies, bins=range(min(accuracies), max(accuracies) + 2), align='left', edgecolor='black', color='skyblue')

plt.title('Human Accuracy')
plt.xlabel('Accuracies')
plt.ylabel('Count')

plt.savefig("IGPT/src/evaluation_methods/graphs/emily_accuracy.png")

categories = ("Eric ICL", "Eric", "Emily ICL", "Emily Fine Tuned", "Emily")
egpt_corr, egpt_inc = 15+11+14+12+9+12+15+8+6+12+10+11+15+5+14, 5+9+5+8+11+8+5+12+14+8+10+9+5+15+6
eog_corr, eog_inc = 11+13+18+1+16+18+12+12+10+9, 9+7+2+19+4+2+8+8+10+11

nb_corr = [9 / 15, 7 / 10, 5 / 9, 7 / 9, 4 / 6]
nb_inc = [6 / 15, 3 / 10, 4 / 9, 2 / 9, 2 / 6]

correct = [egpt_corr / (egpt_corr + egpt_inc), eog_corr / (eog_corr + eog_inc), gpt_corr / (gpt_corr + gpt_inc), bot_corr / (bot_corr + bot_inc), og_corr / (og_corr + og_inc)]
incorrect = [egpt_inc / (egpt_corr + egpt_inc), eog_inc / (eog_corr + eog_inc), gpt_inc / (gpt_corr + gpt_inc), bot_inc / (bot_corr + bot_inc), og_inc / (og_corr + og_inc)]

x = np.arange(len(categories))
bar_width = 0.3
space = 0.05

fig, ax = plt.subplots(figsize=(8, 6))

bar1 = ax.bar(x - (bar_width + space) / 2, correct, bar_width, label='Human Correct', color='lightgreen', edgecolor='black')
bar2 = ax.bar(x - (bar_width + space) / 2, incorrect, bar_width, bottom=correct, label='Human Incorrect', color='lightcoral', edgecolor='black')

bar3 = ax.bar(x + (bar_width + space) / 2, nb_corr, bar_width, label='Naive Bayes Correct', color='darkgreen', edgecolor='black')
bar4 = ax.bar(x + (bar_width + space) / 2, nb_inc, bar_width, bottom=nb_corr, label='Naive Bayes Incorrect', color='darkred', edgecolor='black')

ax.set_xlabel("Categories")
ax.set_ylabel("Question Percentage")
ax.set_title("Human vs Naive Bayes Evaluator Accuracy Distribution")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(title="Accuracy")

plt.tight_layout()
plt.savefig("IGPT/src/evaluation_methods/graphs/category_accuracy.png")

eric_accuracies = [16, 15, 16, 13, 16, 13, 12, 15, 13, 16, 12, 18, 15, 12, 14, 16, 16, 14, 11]

plt.figure(figsize=(8, 6))
plt.hist(eric_accuracies, bins=range(min(eric_accuracies), max(eric_accuracies) + 2), align='left', edgecolor='black', color='skyblue')

plt.title('Human Accuracy')
plt.xlabel('Accuracies')
plt.ylabel('Count')

plt.savefig("IGPT/src/evaluation_methods/graphs/eric_accuracy.png")