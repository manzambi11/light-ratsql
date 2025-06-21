# Light RAT-SQL 🧠

A simplified and optimized variant of Microsoft's RAT-SQL for the text-to-SQL task.  
This version reduces relation complexity while maintaining strong generalization on the Spider dataset.

## 🔍 Overview

**Light RAT-SQL** is based on the original [RAT-SQL](https://github.com/microsoft/rat-sql) architecture.  
It introduces:
- Fewer relation types (reduced from 50+ to 7)
- Improved specialization per attention head
- Faster training and inference with preserved accuracy

This code was developed as part of my PhD research in NLP and semantic parsing (2021–2025).

## 📖 Paper 

📝 [Light RAT-SQL: A RAT-SQL with More Abstraction and Less Embedding of Pre-existing Relations](https://www.texilajournal.com/adminlogin/download.php?category=article&file=Academic_Research_Vol10_Issue2_Article_1.pdf)  
📚 Citation (APA or BibTeX)
NM Ndongala - Texila Int. J. Acad. Res, 2023

## 🚀 Quick Start
1. Follow steps from [RAT-SQL](https://github.com/microsoft/rat-sql)
2. Install spacy  
3. Getting Light RAT-SQL updated files from specific RAT-SQL files:
   
Change of preexisting computation, reducing from 50+ to 7 : 

- \light_ratsql\models\spider\spider_enc_modules.py 

Compute_syntax_dependancy (forward and backward relation) using spacy

- \light_ratsql\models\spider\spider_enc.py 
- \light_ratsql\models\spider\spider_match_utils.py

Spreading relation through heads before Transformer Computation 
- \light_ratsql\models\transformer.py

## Acknowledgment
This project builds upon RAT-SQL by Microsoft Research, licensed under the MIT License.
We thank the original authors for their foundational work in neural semantic parsing.
