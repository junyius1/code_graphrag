# Sample Service

A tiny demonstration repository used by Code GraphRAG tests.

## Overview

The service manages users through `UserService`, which is exercised by the
`cli.py` command line tool and by the HTTP `App` in `app.py`.

## Modules

- `models.py` — `BaseUser`, `User` and `AdminUser` data model with inheritance.
- `services.py` — `UserService` (register/get/count/dump) and
  `NotificationService` (send_email / send_admin_alert).
- `cli.py` — the `main` entry point dispatching register/lookup/notify commands.
- `app.py` — the `App` request router with register/lookup/notify handlers.
- `config.json` — configuration keys: `storage_path`, `max_users`, `default_port`.

## Configuration

Set `storage_path` in `config.json` to control where `UserService` persists
user records.
