import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# List of transactions
transactions = [
    ['22', '36', '11'],
    ['19', '26', '11'],
    ['22', '26'],
    ['23', '22'],
    ['22', '26', '19'],
    ['11', '19'],
    ['31', '11'],
    ['36', '11'],
    ['22', '11'],
    ['19','11']
]

# One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Show the transaction DataFrame
print("Transaksi one-hot:\n", df)

frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)



rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print("Association Rules:\n", rules)

 