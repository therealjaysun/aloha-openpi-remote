# Project troubleshooting

Use the exact recovery printed by the failed command. Raw SSH, Windows, WSL, server, and tunnel output remains in ignored private evidence; do not paste machine identifiers into tracked files.

## SSH alias is not configured

Keep the private `robot-gpu` alias in the Mac user's SSH config, verify the server fingerprint independently, and make one successful batch-key connection. Do not put its hostname, address, username, or key path in `.env` or the repository.

## Windows cannot reach WSL loopback

Leave the policy server on WSL `127.0.0.1`. Do not add a public bind, `netsh portproxy`, firewall rule, relay, or mirrored-network change. Record the bounded route failure and request approval before any networking mutation.

## Mac tunnel port is occupied

Stop the process that legitimately owns `LOCAL_POLICY_PORT` or choose another validated loopback port in the ignored `.env`. The tunnel never signals an unrecorded process.

## Tunnel state is stale or mismatched

Run `make stop`. A valid stale record is removed without signaling; a malformed record, unknown socket, changed PID identity, or non-socket control path fails closed and requires inspection rather than `pkill` or `killall`.

## Policy connection times out

Check the owned WSL server, Windows-loopback route, Mac loopback listener, and exact profile/backend/SHA in that order. Increase a timeout only after measured evidence; never disable it.
