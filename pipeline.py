import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load the data file
data = np.loadtxt("cancer.txt", skiprows=1)

# Split the data into features and attributes
features = data[:, :6]
attributes = data[:, 6:]

to_plot = []

def plot_results(results):
    # Separate the results and benchmarks into two separate lists
    results_list = [result[0] for result in results]
    benchmarks_list = [result[1] for result in results]

    # Create a bar graph with the results and benchmarks
    fig, ax = plt.subplots()
    x_labels = [f'Attribute {i}' for i in range(len(results))]
    bar_width = 0.35
    opacity = 0.8
    rects1 = ax.bar(range(len(results)), results_list, bar_width,
                    alpha=opacity, color='b', label='Results')
    rects2 = ax.bar([i + bar_width for i in range(len(results))], benchmarks_list, bar_width,
                    alpha=opacity, color='r', label='Benchmarks')
    ax.set_xticks([i + bar_width / 2 for i in range(len(results))])
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.show()


# Loop through each attribute and train an SVM to predict it
for i in range(attributes.shape[1]):
    target = attributes[:, i]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    ones = sum(y_train)
    zeros = len(y_train) - ones


    # Train an SVM model on the training data
    # svm = SVC()
    # model = LogisticRegression(class_weight="balanced")
    model = SVC(class_weight="balanced")
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    
    all_ones = np.ones_like(y_pred)
    all_zeros = np.zeros_like(y_pred)


    # Calculate the F1 score and output it
    f1 = f1_score(y_test, y_pred)
    f1_one = f1_score(y_test, all_ones)
    f1_zero = f1_score(y_test, all_zeros)
    to_plot.append((f1, max(f1_zero, f1_one)))
    print("\n\nF1 score for attribute {}: {:.3f}".format(i+1, f1), "\nF1 score for baseline ones {}: {:.3f}".format(i+1, f1_one), "\nF1 score for baseline zeros {}: {:.3f}".format(i+1, f1_zero),)


plot_results(to_plot)

