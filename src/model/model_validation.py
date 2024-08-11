from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def validate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy score: {accuracy:.4f}')

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm)
    print(classification_report(y_test, y_pred))

    return accuracy, y_pred
