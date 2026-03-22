# Test Cases for Cloud Cost Prediction

These test cases cover different GCP services, regions, and usage patterns.  
The values are based on realistic dataset distributions, and expected predictions should fall roughly between **₹80,000 and ₹800,000 INR** depending on service and usage.

---

## Test 1 — Cloud Run (High Traffic)

- **Service:** Cloud Run  
- **Region:** us-central1  
- **Usage Quantity:** 850  
- **Unit:** Requests  
- **Cost per Unit:** 6.50  
- **CPU %:** 78.5  
- **Memory %:** 62.3  
- **Network Inbound:** 420,000,000  
- **Network Outbound:** 380,000,000  
- **Start Date:** 2022-06-15 09:00  
- **End Date:** 2022-06-15 17:00  

---

## Test 2 — BigQuery (Data Processing)

- **Service:** BigQuery  
- **Region:** us-east1  
- **Usage Quantity:** 62  
- **Unit:** Requests  
- **Cost per Unit:** 9.99  
- **CPU %:** 35.2  
- **Memory %:** 88.7  
- **Network Inbound:** 893,000,000,000  
- **Network Outbound:** 893,000,000,000  
- **Start Date:** 2022-03-10 00:00  
- **End Date:** 2022-03-10 09:00  

---

## Test 3 — Cloud Storage (Bulk Storage)

- **Service:** Cloud Storage  
- **Region:** asia-south1  
- **Usage Quantity:** 92.5  
- **Unit:** GB  
- **Cost per Unit:** 3.40  
- **CPU %:** 12.0  
- **Memory %:** 21.0  
- **Network Inbound:** 769,000,000,000  
- **Network Outbound:** 769,000,000,000  
- **Start Date:** 2022-09-01 06:00  
- **End Date:** 2022-09-03 00:00  

---

## Test 4 — Compute Engine (Long Running)

- **Service:** Compute Engine  
- **Region:** europe-west1  
- **Usage Quantity:** 141  
- **Unit:** Hours  
- **Cost per Unit:** 7.83  
- **CPU %:** 9.4  
- **Memory %:** 24.9  
- **Network Inbound:** 461,000,000,000  
- **Network Outbound:** 462,000,000,000  
- **Start Date:** 2022-01-10 10:00  
- **End Date:** 2022-01-17 07:00  

---

## Test 5 — Cloud SQL (Database)

- **Service:** Cloud SQL  
- **Region:** asia-east1  
- **Usage Quantity:** 193  
- **Unit:** Hours  
- **Cost per Unit:** 5.76  
- **CPU %:** 66.0  
- **Memory %:** 97.5  
- **Network Inbound:** 342,000,000,000  
- **Network Outbound:** 343,000,000,000  
- **Start Date:** 2022-01-07 02:00  
- **End Date:** 2022-01-09 21:00  

---

## Notes

- These inputs are aligned with real-world dataset distributions.
- Expected prediction range: **₹80,000 – ₹800,000 INR**
- Suitable for validating ML model performance across:
  - High traffic workloads
  - Data-intensive processing
  - Storage-heavy usage
  - Long-running compute
  - Database workloads