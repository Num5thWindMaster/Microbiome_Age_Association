# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 22:45
# @Author  : HaiqingSun
# @OriginalFileName: bio_test
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

from Bio import Entrez, SeqIO

# 设置邮箱
Entrez.email = "your_email@example.com"

def fetch_species_info(species_name):
    handle = Entrez.esearch(db="taxonomy", term=species_name)
    record = Entrez.read(handle)
    handle.close()

    if record["IdList"]:
        tax_id = record["IdList"][0]
        handle = Entrez.efetch(db="taxonomy", id=tax_id)
        species_info = handle.read()
        handle.close()
        return species_info
    else:
        return None

species_name = "Homo sapiens"
info = fetch_species_info(species_name)

# 保存信息
with open(f"{species_name}.txt", "w") as file:
    file.write(info.decode('utf-8'))


from ete3 import Tree

def fetch_phylogenetic_tree(species_names):
    # 假设你已经得到了物种的NCBI Taxonomy ID
    # 这里使用示例的TaxID，你需要根据实际情况替换
    tax_ids = {
        "Homo sapiens": "9606",
        "Pan troglodytes": "9598",
        "Canis lupus": "9615",
    }

    newick = "(Homo_sapiens:0.5,Pan_troglodytes:0.4,Canis_lupus:0.6);"
    tree = Tree(newick)

    return tree

def visualize_tree(tree):
    tree.render("species_tree.png", w=183, units="mm")
    tree.show()

species_names = ["Homo sapiens", "Pan troglodytes", "Canis lupus"]
# species_info = [fetch_species_info(name) for name in species_names]
phylogenetic_tree = fetch_phylogenetic_tree(species_names)
visualize_tree(phylogenetic_tree)
