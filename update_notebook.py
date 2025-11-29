import json

nb_path = r'c:\Users\Administrator\Desktop\IntelligentMachinesProject1\Project1IM.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Modify Imports
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'from sklearn.decomposition import PCA' in source:
            new_source = source.replace(
                'from sklearn.decomposition import PCA',
                'from sklearn.decomposition import PCA\nfrom sklearn.feature_selection import SelectKBest'
            )
            cell['source'] = new_source.splitlines(keepends=True)
            break

# 2. Modify Outlier Handling
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def cap_outliers(data, columns):' in source and 'X_train_capped = cap_outliers(X_train_num, num_cols)' in source:
            # We want to exclude Rainfall from capping
            new_source = source.replace(
                'X_train_num = X_train[num_cols]\nX_train_capped = cap_outliers(X_train_num, num_cols)',
                "cols_to_cap = [c for c in num_cols if c != 'Rainfall']\n"
                "X_train_num = X_train[cols_to_cap]\n"
                "X_train_capped = cap_outliers(X_train_num, cols_to_cap)"
            )
            cell['source'] = new_source.splitlines(keepends=True)
            break

# 3. Modify PCA to include Feature Selection
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'pca = PCA(n_components=0.95)' in source:
            # Add Feature Selection before PCA
            new_code = [
                "# Feature Selection using Mutual Information\n",
                "selector = SelectKBest(mutual_info_classif, k=30)\n",
                "X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)\n",
                "X_test_selected = selector.transform(X_test_df)\n",
                "print(f\"Selected Feature Count: {X_train_selected.shape[1]}\")\n",
                "\n",
                "pca = PCA(n_components=0.95) # Retain 95% variance\n",
                "X_train_pca = pca.fit_transform(X_train_selected)\n",
                "X_test_pca = pca.transform(X_test_selected)\n",
                "\n",
                "print(f\"Original Feature Count: {X_train_balanced.shape[1]}\")\n",
                "print(f\"PCA Feature Count: {X_train_pca.shape[1]}\")\n",
                "\n",
                "# Visualization of Variance\n",
                "plt.figure(figsize=(10,6))\n",
                "plt.plot(np.cumsum(pca.explained_variance_ratio_))\n",
                "plt.xlabel('Number of Components')\n",
                "plt.ylabel('Cumulative Explained Variance')\n",
                "plt.title('PCA Explained Variance Ratio')\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
            cell['source'] = new_code
            break

# 4. Modify Model Tuning
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'voting_clf = VotingClassifier' in source:
            # Expand params
            new_source = source.replace(
                "'rf__n_estimators': [100, 200],\n    'rf__max_depth': [10, 20],\n    'gb__n_estimators': [100, 200],\n    'gb__learning_rate': [0.1]",
                "'rf__n_estimators': [200, 300],\n    'rf__max_depth': [20, 30, None],\n    'rf__min_samples_leaf': [1, 2],\n    'gb__n_estimators': [200, 300],\n    'gb__learning_rate': [0.05, 0.1, 0.2],\n    'gb__max_depth': [3, 5]"
            )
            cell['source'] = new_source.splitlines(keepends=True)
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
