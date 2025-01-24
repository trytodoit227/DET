import json
import pandas as pd
from scipy.stats import pearsonr
import krippendorff



def calculate_relevance(human_data, machine_data, exp_name):
    # Ensure that comparisons are made for the same text entries
    human_texts = [entry["original_data"]["text"] for entry in human_data]

    human_scores = []
    machine_scores = []

    # Extract the EQ scores for the same texts with the same order
    for text in human_texts:
        for entry in human_data:
            if entry["original_data"]["text"] == text:
                human_scores.append(int(entry["evaluating_result"]["score_clarity"]))
                human_scores.append(int(entry["evaluating_result"]["score_relevance"]))
                human_scores.append(int(entry["evaluating_result"]["score_completeness"]))
                human_scores.append(int(entry["evaluating_result"]["score_consistency"]))
                human_scores.append(int(entry["evaluating_result"]["score_credibility"]))
                break
        for entry in machine_data:
            if entry["original_data"]["text"] == text:
                machine_scores.append(int(entry["evaluating_result"]["score_clarity"]))
                machine_scores.append(int(entry["evaluating_result"]["score_relevance"]))
                machine_scores.append(int(entry["evaluating_result"]["score_completeness"]))
                machine_scores.append(int(entry["evaluating_result"]["score_consistency"]))
                machine_scores.append(int(entry["evaluating_result"]["score_credibility"]))
                break


    
    data_matrix = pd.DataFrame({
        'Human scores': human_scores,
        'Machine scores': machine_scores
    })

    # save the data matrix to a csv file
    data_matrix.to_csv(f'{exp_name}_data_matrix.csv', index=False)

    data_matrix_np = data_matrix.to_numpy().T

    # Calculate Pearson correlation coefficient
    pearson_corr, pearson_p_value = pearsonr(human_scores, machine_scores)
    # Calculate Krippendorff’s alpha 
    krippendorff_alpha = krippendorff.alpha(reliability_data=data_matrix_np, level_of_measurement='ordinal')

    return pearson_corr, pearson_p_value, krippendorff_alpha

# Load the data from the JSON files
with open('Human_evaluated_results_120_samples.json') as f:
    human_data = json.load(f)["results"]

with open('Aggregated_machine_samples_gpt-4o.json') as f:
    machine_data = json.load(f)["results"]

# Calculate the relevance metrics
exp_name = "human_vs_gpt-4o"
pearson_corr_gpt, pearson_p_value_gpt, krippendorff_alpha_gpt = calculate_relevance(human_data, machine_data, exp_name)

print(f"Pearson correlation coefficient for gpt-4o: {pearson_corr_gpt}")
print(f"P-value for the Pearson correlation coefficient for gpt-4o: {pearson_p_value_gpt}")
print(f"Krippendorff’s alpha for gpt-4o: {krippendorff_alpha_gpt}")


# Load the data from the JSON files
with open('Human_evaluated_results_100_samples.json') as f:
    human_data = json.load(f)["results"]

with open('Aggregated_machine_samples_claude.json') as f:
    machine_data = json.load(f)["results"]

# Calculate the relevance metrics
exp_name = "human_vs_claude3.5"
pearson_corr_claude, pearson_p_value_claude, krippendorff_alpha_claude = calculate_relevance(human_data, machine_data, exp_name)

print(f"Pearson correlation coefficient for claude3.5: {pearson_corr_claude}")
print(f"P-value for the Pearson correlation coefficient for claude3.5: {pearson_p_value_claude}")
print(f"Krippendorff’s alpha for claude3.5: {krippendorff_alpha_claude}")

# save pearson_corr, pearson_p_value, krippendorff_alpha results to csv columns file, keep two decimal places
result = pd.DataFrame({
    'Metrics': ['Human VS GPT-4o', 'Human VS Claude 3.5-sonnet'],
    'Pearson correlation coefficient': [round(pearson_corr_gpt, 3), round(pearson_corr_claude, 3)],
    'Pearson P-value': [f"{pearson_p_value_gpt:.3e}", f"{pearson_p_value_claude:.3e}"],
    'Krippendorff’s alpha': [round(krippendorff_alpha_gpt, 3), round(krippendorff_alpha_claude, 3)]
})

#  save as csv
result.to_csv('relevance_metrics.csv', index=False)


