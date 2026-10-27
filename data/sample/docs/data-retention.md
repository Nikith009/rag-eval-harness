# Data Retention and Deletion

This policy sets how long Acme Cloud keeps different kinds of data and how data is deleted. Keeping data longer than needed increases risk, so the default is to delete data once its retention period ends.

## Retention periods

- Customer application data is kept for as long as the customer's contract is active. After a contract ends, the data is retained for 30 days so the customer can export it, then permanently deleted within the following 60 days.
- Backups of production databases are kept for 35 days and are encrypted with keys managed in the cloud key management service.
- Application logs are kept for 90 days. Security audit logs are kept for one year because regulators and auditors may ask for them.
- Support tickets are kept for three years after they are closed.
- Financial records, such as invoices and payroll data, are kept for seven years to meet tax requirements.
- Candidate data from hiring is deleted 12 months after the hiring decision unless the candidate agrees to be kept on file longer.

## Deletion requests

Customers and end users can request deletion of their personal data. Deletion requests must be completed within 30 days of being verified. Data in backups is not edited; instead it expires naturally when the backup reaches the end of its 35-day retention period.

## Data classification

Data is classified into four levels: public, internal, confidential and restricted. Customer data and credentials are always restricted. Restricted data must be encrypted in transit and at rest, must not be copied to personal devices, and must never be pasted into external tools that have not passed a security review, including AI assistants.

## Test environments

Production customer data must not be used in development or staging environments. Teams that need realistic data use the synthetic data generator or a dataset anonymized and approved by the data protection officer.

## Ownership

Each data store has a named owner who is responsible for applying the retention rules. Owners review their data stores every six months and confirm that automated deletion jobs are running.
