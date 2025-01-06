import pandas as pd


def extract_direct_recommendations(AB_df, BA_df):
    results = {
        'dataSI': [],
        'recommendations': []
    }

    # Iterate through each column (SI)
    for column in AB_df.columns:
        # Get items where AB similarity is above threshold
        filter_top1 = AB_df[column].nlargest(15)
        top1 = filter_top1[filter_top1 > 0.25]
        # Get corresponding BA values, limited to available items
        top2 = BA_df[column].reindex(top1.index)
        # Sort by BA values and take up to 7 items
        top2 = top2.nlargest(min(7, len(top2)))
        # Reindex top1 to match top2's indices
        top1 = top1.reindex(top2.index)

        recommendations = []
        for idx in range(len(top2)):
            recommendations.append({
                'MK Rekomendasi': top1.index[idx],
                'P(B|A)': float(top1.iloc[idx]),
                'P(A|B)': float(top2.iloc[idx])
            })

        results['dataSI'].append(column)
        results['recommendations'].append(recommendations)
    
    # Add tie-breaking logic for identical P(A|B) values
    for i, recommendations in enumerate(results['recommendations']):
        # Convert the recommendations to a DataFrame for processing
        df_recommendations = pd.DataFrame(recommendations)

        # Apply tie-breaking logic: sort first by P(A|B), then break ties by P(B|A)
        df_recommendations = df_recommendations.sort_values(
            by=['P(A|B)', 'P(B|A)'],  # Primary sort by P(A|B), secondary by P(B|A) for ties
            ascending=[False, False]  # Both in descending order
        )

        # Update the recommendations with the tie-broken order
        results['recommendations'][i] = df_recommendations.to_dict('records')

    return pd.DataFrame(results)
