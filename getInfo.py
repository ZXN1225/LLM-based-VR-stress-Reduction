import chromadb
import pandas as pd
# 1. 加载之前保存的数据库
db_client = chromadb.PersistentClient(path="./PictureBase")
collection = db_client.get_collection(name="nature_environments")

# 2. 查看数据库中存储的总条数
count = collection.count()
print(f"数据库中共有 {count} 条记录")

# 3. 获取所有数据（包括文档内容和元数据）
results = collection.get(
    include=["documents", "metadatas"]
)

# 4. 打印前几条数据进行查看
for i in range(min(5, count)):
    print(f"--- 记录 {i+1} ---")
    print(f"文件名: {results['metadatas'][i]['filename']}")
    print(f"描述内容: {results['documents'][i][:100]}...")
    print(f"亮度/色温: {results['metadatas'][i]['brightness']} / {results['metadatas'][i]['temp_tendency']}")
    print(f"对比度: {results['metadatas'][i]['contrast']}")
    print("\n")

results = collection.get(include=["documents", "metadatas"])

# 整理成列表格式
data_list = []
for i in range(len(results['ids'])):
    entry = {
        "ID": results['ids'][i],
        "Caption": results['documents'][i],
        "Filename": results['metadatas'][i]['filename'],
        "Brightness": results['metadatas'][i]['brightness'],
        "Contrast": results['metadatas'][i]['contrast'],
        "Temp_Tendency": results['metadatas'][i]['temp_tendency'],
        "Sharpness": results['metadatas'][i]['sharpness'],
        "Colorfulness": results['metadatas'][i]['colorfulness'],
        "Environment": results['metadatas'][i]['environment'],
        "Mood Tags": results['metadatas'][i]['mood'],
        "Objects": results['metadatas'][i]['objects'],
        "Estimated Kelvin": results['metadatas'][i]['Estimated_kelvin'],
        "is_optimized_target": results['metadatas'][i]['is_optimized_target'],
    }
    data_list.append(entry)

# 转为 DataFrame 并保存
df = pd.DataFrame(data_list)
df.to_csv("database_export.csv", index=False)
print("数据已导出至 database_export.csv")