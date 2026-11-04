# Build the dashboard, then the Python service that serves it.
FROM node:22-alpine AS dashboard
WORKDIR /src/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/ /app/backend/
COPY --from=dashboard /src/backend/regressionlens/api/static /app/backend/regressionlens/api/static
RUN pip install --no-cache-dir "/app/backend[server]" && useradd --create-home rlens
USER rlens
EXPOSE 8080
CMD ["rlens", "server", "--host", "0.0.0.0", "--port", "8080"]
