# Engineering Onboarding

This guide covers the first month for new engineers at Acme Cloud. Each new engineer is paired with an onboarding buddy from their team, who is the first person to ask about code, tools and team habits.

## Week one

On day one you receive your laptop, accounts for GitHub, Slack, Linear and the cloud console, and access to the internal developer portal. Access to production systems is not granted in the first week. The goal for week one is to set up the local development environment, read the architecture overview, and ship a small change to a staging environment.

## Development workflow

All code lives in GitHub. Every change goes through a pull request that needs at least one approving review from a teammate; changes to payment or authentication code need two approvals, one of them from a code owner. Continuous integration runs on GitHub Actions and must pass before merging. The main branch is deployed to staging automatically after every merge, and production deploys happen through the deploy pipeline on weekdays between 9:00 and 16:00 Central Time. Production deploys are frozen on Fridays after 12:00 and during the winter break.

## Production access

Engineers can request read-only production access after completing the security training, usually during week two. Write access to production databases is granted only to on-call engineers, through short-lived credentials that expire after eight hours.

## On-call

Engineers join the on-call rotation after their third month. A shift lasts one week, from Monday 10:00 to the next Monday 10:00. On-call engineers receive a stipend of 400 USD per week of on-call, and an additional day off if they were paged more than three times outside working hours during the shift.

## Coding standards

Backend services are written in Python and Go; the web frontend uses TypeScript and React. Every service must expose health checks and structured JSON logs, and every new endpoint needs tests. Architecture decisions that affect more than one team are written up as a short design document and reviewed in the weekly architecture forum on Thursdays.

## First month goals

By the end of the first month a new engineer should have merged at least five pull requests, joined one incident review as an observer, and presented a short walkthrough of something they learned to their team.
