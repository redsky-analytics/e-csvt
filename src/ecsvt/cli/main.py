#%%
from ecsvt import HybridEntityResolver

#%%
def main():
    resolver = HybridEntityResolver()
    result = resolver.find_duplicates(
        'sample_data.csv',
        threshold=0.75
    )
    print(result.columns)

if __name__ == "__main__":
    main()
