def categorize(pdf,rownames):
    df = pdf
    for rowname in rownames:
        df[rowname] = str(df[rowname])

    return df

