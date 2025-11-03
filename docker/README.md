# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Building and Running the Application

To build and run the application, you will use the `docker-compose.yml` file located in the root of the project. This file is configured to use the `Dockerfile.non_root` for a secure, non-root container environment.

### 1. Set the Master Key

The application requires a `MASTER_KEY` for signing and validating tokens. You must set this key as an environment variable before running the application.

Create a `.env` file in the root of the project and add the following line:

```
MASTER_KEY=your-secret-key
```

Replace `your-secret-key` with a strong, randomly generated secret.

### 2. Build and Run the Containers

Once you have set the `MASTER_KEY`, you can build and run the containers using the following command:

```bash
docker compose up -d --build
```

This command will:

-   Build the Docker image using `Dockerfile.non_root`.
-   Start the `litellm`, `litellm_db`, and `prometheus` services in detached mode (`-d`).
-   The `--build` flag ensures that the image is rebuilt if there are any changes to the Dockerfile or the application code.

### 3. Verifying the Application is Running

You can check the status of the running containers with the following command:

```bash
docker compose ps
```

To view the logs of the `litellm` container, run:

```bash
docker compose logs -f litellm
```

### 4. Stopping the Application

To stop the running containers, use the following command:

```bash
docker compose down
```

## Testing Prisma Offline Mode (Air-Gapped Environments)

The LiteLLM Docker image supports running in air-gapped/restricted network environments by pre-caching Prisma binaries at build time. This feature is enabled by default.

### Testing with Restricted Network Access

**Platform Support:**
- **AMD64 (x86_64)**: Full offline mode support - all Prisma binaries are pre-cached at build time
- **ARM64 (Apple Silicon, ARM servers)**: Query-engine is pre-cached, but schema-engine may require runtime download on first migration
  - **Workaround**: On ARM64, if you encounter schema-engine errors, allow network access during first startup or set `PRISMA_OFFLINE_MODE=false`

To test that the container works without outbound network access:

#### Option 1: Using docker-compose with restricted network

Create a `docker-compose.offline-test.yml` file:

```yaml
services:
  litellm:
    build:
      context: .
      dockerfile: docker/Dockerfile.non_root
    image: litellm-offline-test:latest
    networks:
      - isolated
    ports:
      - "4000:4000"
    environment:
      DATABASE_URL: "postgresql://llmproxy:dbpassword9090@db:5432/litellm"
      STORE_MODEL_IN_DB: "True"
      PRISMA_OFFLINE_MODE: "true"  # Explicitly enable offline mode (default)
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: postgres:16
    restart: always
    networks:
      - isolated
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: llmproxy
      POSTGRES_PASSWORD: dbpassword9090
    volumes:
      - postgres_data_test:/var/lib/postgresql/data

networks:
  isolated:
    driver: bridge
    internal: true  # No external network access

volumes:
  postgres_data_test:
```

Then run:

```bash
# Build and start with restricted network
docker compose -f docker-compose.offline-test.yml up -d --build

# Check logs - should see "Using cached Prisma CLI"
docker compose -f docker-compose.offline-test.yml logs -f litellm

# Verify no outbound network attempts
docker compose -f docker-compose.offline-test.yml exec litellm sh -c "wget -T 5 google.com 2>&1 || echo 'No network access - expected'"

# Clean up
docker compose -f docker-compose.offline-test.yml down -v
```

**ARM64 workaround (if schema-engine errors occur):**
```bash
# Option A: Allow network access for first startup, then restrict
docker compose -f docker-compose.offline-test.yml up -d  # Let schema-engine download
docker compose -f docker-compose.offline-test.yml down
# Now edit docker-compose to add "internal: true" to network

# Option B: Disable offline mode for ARM64
PRISMA_OFFLINE_MODE=false docker compose -f docker-compose.offline-test.yml up -d --build
```

#### Option 2: Using plain Docker with network disabled

```bash
# Build the image first
cd /path/to/litellm-fork
docker build -f docker/Dockerfile.non_root -t litellm-offline-test .

# Run with network completely disabled
docker run --network none \
  -e DATABASE_URL="sqlite:///litellm.db" \
  -e MASTER_KEY="test-key-123" \
  litellm-offline-test

# Container should start successfully without network
```

### Verifying Pre-cached Binaries

To verify that Prisma binaries are pre-cached in the image:

```bash
# Check pre-cached Prisma CLI
docker run --rm litellm-offline-test ls -la /app/.cache/prisma-python/binaries/node_modules/.bin/

# Should show the prisma executable

# Check Prisma engine binaries
docker run --rm litellm-offline-test ls -la /app/.cache/prisma-python/binaries/engines/

# Should show query engine, schema engine, etc.
```

### Environment Variables for Offline Mode

The following environment variables control Prisma offline mode (set by default in Dockerfile):

- `PRISMA_OFFLINE_MODE=true` - Enable offline mode (uses pre-cached binaries)
- `PRISMA_BINARY_CACHE_DIR=/app/.cache/prisma-python/binaries` - Location of cached binaries
- `NPM_CONFIG_PREFER_OFFLINE=true` - Prevent npm from attempting downloads
- `PRISMA_CLI_PATH` - (Optional) Custom path to Prisma CLI if not using default location

To disable offline mode and allow runtime downloads (online environments):

```yaml
environment:
  PRISMA_OFFLINE_MODE: "false"
```

## Troubleshooting

-   **`build_admin_ui.sh: not found`**: This error can occur if the Docker build context is not set correctly. Ensure that you are running the `docker-compose` command from the root of the project.
-   **`Master key is not initialized`**: This error means the `MASTER_key` environment variable is not set. Make sure you have created a `.env` file in the project root with the `MASTER_KEY` defined.
-   **`Prisma CLI not found`**: If you see warnings about Prisma CLI not being found, verify that the Docker build completed successfully and check `/app/.cache/prisma-python/binaries` exists in the container.
-   **Network errors during startup**: If you see network errors despite offline mode being enabled, check that `PRISMA_OFFLINE_MODE=true` is set and that the binaries were cached during build time.
