import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

def evaluate_model_cv(model, X, y, groups=None, n_splits=5, random_seed=42):
    """
    Evaluates a machine learning pipeline using robust cross-validation.
    Optimized to compute the Macro-F1 score as required by the Kaggle competition.
    """
    # Create the custom Macro-F1 scorer to match the assignment evaluation
    macro_f1_scorer = make_scorer(f1_score, average='macro')

    # Determine the Cross-Validation Strategy
    if groups is not None:
        # Use GroupKFold if subject IDs are known to ensure unseen subjects generalize
        print(f"Evaluating using GroupKFold (splits={n_splits})...")
        cv_strategy = GroupKFold(n_splits=n_splits)
        # Groups parameter must be passed to cross_val_score
        scores = cross_val_score(model, X, y, groups=groups, cv=cv_strategy, scoring=macro_f1_scorer, n_jobs=-1)
        
    else:
        # Use StratifiedKFold to maintain class balances across the 6 physical actions
        print(f"Evaluating using StratifiedKFold (splits={n_splits})...")
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=macro_f1_scorer, n_jobs=-1)

    # Output the results
    print("-" * 30)
    print(f"Fold Macro-F1 Scores: {np.round(scores, 4)}")
    print(f"Mean Macro-F1 Score:  {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print("-" * 30)

    return scores