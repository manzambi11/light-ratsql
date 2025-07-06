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

- NM Ndongala. (2023) Light RAT-SQL: A RAT-SQL with More Abstraction and Less Embedding of Pre-existing Relations. Texila International Journal of Academic Research, 10, 1–11. https://doi.org/10.21522/tijar.2014.10.02.art001

## 🚀 Quick Start

- **Docker command** : docker pull manzambi11/light_ratsql:latest

- **or follow these steps:**

<ol>
      <li><p>Follow steps from <a href="https://github.com/microsoft/rat-sql">[RAT-SQL]</a></p></li>
      <li> Install spacy 3.4: 
      <ul>
      <li><code>pip install spacy==3.4</code></li>
      <li><code>python -m spacy download en_core_web_sm</code></li>
      </ul>
      </li>
      <li>Getting Light RAT-SQL updated files from specific RAT-SQL files: 
      <ul>
      <li> Change of preexisting computation, reducing from 50+ to 7 :
      <li>\light_ratsql\models\spider\spider_enc_modules.py </li>
      </ul>
      </li>
      <li> Compute_syntax_dependancy (forward and backward relation) using spacy
      <ul>
      <li>\light_ratsql\models\spider\spider_enc.py</li>
      <li>\light_ratsql\models\spider\spider_match_utils.py</li>
      </ul>      
      </li>
      <li> Spreading relation through heads before Transformer Computation <br> 
      <ul>
      <li>\light_ratsql\models\transformer.py</li>
      </ul>
      </li>
    </li>
</ol>

## Author

- [Nathan MANZAMBI NDONGALA, Ph.D in CS](https://www.linkedin.com/in/nathan-manzambi-59a2285b/)

## Acknowledgment
This project builds upon RAT-SQL by Microsoft Research, licensed under the MIT License.
We thank the original authors for their foundational work in neural semantic parsing.
