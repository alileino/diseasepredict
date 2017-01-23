# diseasepredict
 A project for a machine learning course. It scrapes data about symptom - disease links and predicts diseases based on symptoms.

# Criticism of the project
The problem formulation given in the course did not include any validation/test set, and none were publicly available. That's why for each disease there is exactly one training example, which means that each class has exactly one training example. This makes validation and testing from hard to impossible - there is no clear way to measure the accuracy.

The approach taken to remedy the missing test set was to generate it using various distributions. It is not clear if these have any significance in real-life applications.
