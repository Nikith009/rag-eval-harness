# Security Incident Response

A security incident is any event that may compromise the confidentiality, integrity or availability of Acme Cloud systems or customer data. Examples include a phishing email that someone clicked, a lost laptop, leaked credentials, malware, or unusual access to production databases.

## How to report

Report suspected incidents immediately in the #security-incidents Slack channel or by email to security@acmecloud.example. Outside business hours, page the on-call security engineer through PagerDuty. Lost or stolen devices must be reported within one hour. Do not try to investigate on your own, and do not delete suspicious emails, because the security team needs them as evidence.

## Severity levels

- SEV1: confirmed exposure of customer data or a full outage of a production service. The incident commander is paged immediately, and the executive team is notified within 30 minutes.
- SEV2: a likely compromise without confirmed customer impact, such as leaked production credentials. Response starts within one hour.
- SEV3: a contained issue with low risk, such as a single phishing click with no credentials entered. Response starts within one business day.

## Response steps

1. Triage: the on-call engineer confirms the incident, assigns a severity and opens an incident ticket.
2. Contain: isolate affected systems, revoke exposed credentials and block malicious accounts or IP addresses.
3. Eradicate: remove the root cause, for example by patching a vulnerability or reimaging a device.
4. Recover: restore services from known-good backups and monitor closely for 72 hours.
5. Review: hold a blameless postmortem within five business days of resolution.

## Customer notification

If customer data is affected, the legal team decides whether customers and regulators must be notified. Under our customer agreements, affected customers are notified within 72 hours of confirming a data breach. Only the legal and communications teams may speak to customers or the press about an incident.

## Credential leaks

If a secret such as an API key or password is committed to a repository, rotate it right away, even if the repository is private. Removing the commit is not enough, because the secret may already be cached or cloned. Production secrets live only in the company secrets manager and are rotated every 90 days.
