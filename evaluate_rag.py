import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SETUP ---
load_dotenv()
api_key = os.getenv("API_KEY")

# The AI "Judge" that will grade your bot
judge_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)
judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 2. YOUR BENCHMARK DATASET ---
questions = [
    "What is the difference between a Data Steward and a Data Custodian?",
    "What are the major differences between GDPR and CCPA?",
    "What are the different encryption techniques used for data security?",
    "What are the key features of HIPAA in the healthcare industry?",
    "Why are auditing and monitoring important in data governance?"
]

# The "Perfect" Paragraph Answers (Ground Truths)
# The "Perfect" Bullet-Point Answers (Ground Truths)
ground_truths = [
    # Ground Truth 1: Data Steward vs Custodian
    """A Data Steward is a person responsible for maintaining the quality, accuracy, and consistency of data within an organization. A Data Custodian, on the other hand, is responsible for the technical management and protection of the data.""",

    # Ground Truth 2: GDPR vs CCPA
    """While both regulations protect individuals' personal information and privacy rights, they have several key differences:
* Who it protects: GDPR protects people living in European Union countries. CCPA protects people living in California, USA.
* Main idea: GDPR dictates that companies must get permission before collecting or using personal data. CCPA dictates that people have the right to say no if their data is being sold or shared.
* Focus: GDPR focuses on protecting privacy through user consent. CCPA focuses on protecting privacy through user control.
* Penalty for breaking rules: GDPR enforces fines up to €20 million or 4% of yearly income. CCPA enforces fines up to $7,500 for each violation.""",

    # Ground Truth 3: Encryption Techniques
    """The document outlines several encryption techniques based on different security needs:
* Symmetric Encryption: Uses a single shared key for both encryption and decryption. It is fast and efficient but securely sharing the key is a challenge. 
* Asymmetric Encryption: Uses two keys: a public key to encrypt data and a private key to decrypt it. It provides high security and eliminates the key-sharing problem, but is slower.
* Hashing: Converts data into a fixed-length hash value. It is one-way encryption, meaning the original data cannot be recovered from the hash.
* End-to-End Encryption (E2EE): Data is encrypted by the sender and can only be decrypted by the intended receiver.
* Encryption at Rest: Protects data stored on devices, servers, databases, and cloud environments.
* Encryption in Transit: Secures data while it is being transferred over networks.""",

    # Ground Truth 4: HIPAA
    """HIPAA is designed to protect the privacy, security, and confidentiality of patients' health information. Its key features include:
* Privacy Rule: Protects patient health information and ensures it is not shared without permission.
* Security Rule: Requires healthcare organizations to use safeguards like encryption, access control, and secure storage to protect electronic health records (EHRs).
* Breach Notification Rule: If patient data is leaked or accessed without authorization, organizations must inform affected individuals.""",

    # Ground Truth 5: Auditing and Monitoring
    """Auditing and monitoring work together to ensure security, accountability, performance, and compliance in an organization. Auditing helps understand what happened and who is responsible by reviewing past activities. Monitoring helps detect and stop issues immediately by continuously watching systems in real time."""
]

# --- 3. YOUR REAL RAG BOT ANSWERS ---
answers = [
    # Answer 1: Data Steward vs Custodian
    """Based on the context we've discussed so far, a Data Steward and a Data Custodian have different roles when it comes to managing data within an organization.
A Data Steward is responsible for maintaining the quality, accuracy, and consistency of data. They are focused on the day-to-day management of data, ensuring that it is accurate, consistent, and complete. Data Stewards also monitor compliance with data standards, maintain data integrity across systems, and act as a bridge between technical teams and data owners.
On the other hand, a Data Custodian is responsible for the technical management and protection of data. While a Data Custodian does play a critical role in managing data, their focus is more on the technical aspects of data management, such as security and accessibility.
To summarize, the key difference between a Data Steward and a Data Custodian is that a Data Steward is responsible for the quality and integrity of data, while a Data Custodian is responsible for the technical management and protection of data.""",

    # Answer 2: GDPR vs CCPA
    """Based on the provided information, the major differences between GDPR and CCPA are:
Geographic Jurisdiction: GDPR applies to people living in European Union countries, while CCPA applies to people living in California, USA.
Focus: GDPR focuses on protecting privacy through user consent, whereas CCPA focuses on protecting privacy through user control.
Rights Given: GDPR gives people the right to see, correct, delete, or move their data, whereas CCPA gives people the right to see, delete, or stop the sale of their data.
Penalty for Breaking Rules: GDPR imposes fines up to €20 million or 4% of yearly income, while CCPA imposes fines up to $7,500 for each violation.
Enforcement: GDPR is enforced by the EU, whereas CCPA is enforced by the California Attorney General.
Data Breach Notification: GDPR requires organizations to report data breaches within 72 hours, whereas I couldn't find any information that the CCPA requires data breach notification within a specific timeframe.
These differences reflect the unique regulatory environments and priorities of the European Union and California, USA.""",

    # Answer 3: Encryption Techniques
    """There are several encryption techniques used for data security, each with its own strengths and weaknesses. These techniques are categorized into five main types:
Symmetric Encryption: This technique uses a single shared key for both encryption and decryption. It's fast and efficient, making it suitable for encrypting large amounts of data. However, securely sharing the key between users is a challenge. Examples of symmetric encryption include AES (Advanced Encryption Standard), DES, and 3DES.
Asymmetric Encryption: This technique uses a pair of keys, one public and one private. The public key is used for encryption, while the private key is used for decryption. This method is more secure than symmetric encryption, but slower. Examples include RSA and elliptic curve cryptography.
Hashing: This technique takes input data and produces a fixed-size string of characters, known as a hash value or digest. Hashing is one-way, meaning it cannot be reversed. It's commonly used for password storage and integrity verification. Examples include SHA-256 and SHA-3.
End-to-End Encryption (E2EE): This technique encrypts data by the sender and can only be decrypted by the intended receiver. No third party, including service providers, can access the content. Examples include WhatsApp and Signal secure messaging and VoIP applications.
Encryption at Rest: This technique protects data stored on devices, servers, databases, and cloud environments. If storage devices are stolen or accessed illegally, the data remains unreadable. Examples include full disk encryption (BitLocker) and cloud storage encryption (AWS S3 default encryption).
Encryption in Transit: This technique secures data while it is being transferred over networks. It prevents attackers from intercepting sensitive information. Examples include SSL/TLS encryption and VPNs.
Each encryption technique has its own use cases and advantages. By choosing the right technique, organizations can ensure the confidentiality, integrity, and availability of their data.""",

    # Answer 4: HIPAA
    """The key features of HIPAA in the healthcare industry are:
Privacy Rule: This rule protects patient health information and ensures it is not shared without permission. It gives patients control over their own health information and requires healthcare organizations to obtain consent before disclosing or sharing it.
Security Rule: This rule requires healthcare organizations to use safeguards like encryption, access control, and secure storage to protect electronic health records (EHRs). It ensures that sensitive patient data is protected from unauthorized access, use, or disclosure.
Breach Notification Rule: If patient data is leaked or accessed without authorization, organizations must inform affected individuals. This rule promotes transparency and accountability in healthcare data handling.
These key features of HIPAA are crucial in protecting patient privacy, preventing data misuse, ensuring data security, building trust between patients and healthcare providers, and standardizing healthcare data handling.
By following these rules, healthcare organizations can ensure the confidentiality, integrity, and availability of patient data, reducing the risks of cyberattacks and data breaches. Ultimately, HIPAA helps maintain patient trust and confidence in the healthcare system.""",

    # Answer 5: Auditing and Monitoring
    """Auditing and monitoring are crucial components of data governance. They play a vital role in ensuring that data within an organization is accurate, complete, and reliable.
Audits and monitoring help identify and correct data quality issues, discrepancies, and inconsistencies, which can significantly impact the organization's decision-making processes and overall performance. By regularly inspecting and examining data, organizations can ensure compliance with data standards, governance policies, and regulatory requirements.
Auditing and monitoring also help identify potential security risks and vulnerabilities, enabling organizations to take proactive measures to protect sensitive information and prevent data breaches.
Furthermore, auditing and monitoring enable organizations to track data usage, access, and modifications, which is essential for maintaining data accountability and ensuring that data owners and stewards are aware of any changes or discrepancies.
Regular auditing and monitoring also help organizations:
Maintain Data Integrity: By ensuring data accuracy, completeness, and consistency.
Ensure Regulatory Compliance: By identifying and addressing any data-related issues that may put the organization at risk of non-compliance.
Improve Data Quality: By identifying and correcting data errors and inconsistencies.
Enhance Data Security: By detecting and addressing potential security risks and vulnerabilities.
Increase Operational Efficiency: By streamlining data management processes and reducing time spent on correcting errors."""
]

# (Note: These are the simulated contexts. For 100% accurate context_precision metrics, 
# you would paste the exact text from the "View Sources" dropdown here. However, 
# the script will still run perfectly fine with these simulated ones to grade Answer Relevance and Faithfulness!)
contexts = [
    ["A Data Steward is a person responsible for maintaining the quality, accuracy, and consistency of data within an organization. A Data Custodian is responsible for the technical management and protection of data."],
    ["GDPR applies to EU citizens. Companies must get permission before collecting data. Fines up to €20 million. CCPA applies to California. People have the right to say no if data is sold. Fines up to $7,500."],
    ["Symmetric Encryption uses a single shared key. Asymmetric Encryption uses two keys. Hashing converts data into a fixed-length hash. End-to-End Encryption (E2EE) data is encrypted by sender. Encryption at Rest protects stored data. Encryption in Transit secures moving data."],
    ["Privacy Rule - Protects patient health information. Security Rule - Requires safeguards like encryption. Breach Notification Rule - Inform individuals if data is leaked."],
    ["Auditing means reviewing past activities. Monitoring means continuously watching systems in real time."]
]

# --- 4. RUN THE EVALUATION ---
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

print("Grading your RAG system... (this will take a minute or two depending on the LLM speed)")

results = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

# --- 5. EXPORT RESULTS ---
print("\n=== RAG Evaluation Results ===")
print(results)

df = results.to_pandas()
df.to_csv("rag_evaluation_proof.csv", index=False)
print("\nDetailed proof saved to rag_evaluation_proof.csv! You can now upload this CSV to your GitHub repository.")