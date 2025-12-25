# -*- coding: utf-8 -*-
from src.data_loader import load_all_documents
import pprint

print("=" * 60)
print("INSPECTING DOCUMENT OBJECTS")
print("=" * 60)

docs = load_all_documents("data")

print("\nTotal dokumen:", len(docs))

if len(docs) > 0:
    first_doc = docs[0]
    print("Tipe:", type(first_doc).__name__)
    
    print("\n" + "=" * 60)
    print("STRUKTUR DOCUMENT OBJECT")
    print("=" * 60)
    
    print("\nAttributes (non-private):")
    attrs = [a for a in dir(first_doc) if not a.startswith("_")]
    print(attrs)
    
    print("\npage_content (100 char):")
    print(repr(first_doc.page_content[:100]))
    
    print("\nmetadata:")
    pprint.pprint(first_doc.metadata)
    
    print("\n" + "=" * 60)
    print("PREVIEW", min(3, len(docs)), "DOKUMEN")
    print("=" * 60)
    
    for i, doc in enumerate(docs[:3]):
        print("\n--- Dokumen", i+1, "---")
        print("Source:", doc.metadata.get("source", "N/A"))
        print("Length:", len(doc.page_content), "chars")
        print("Preview:", doc.page_content[:80], "...")
        print("Metadata keys:", list(doc.metadata.keys()))
