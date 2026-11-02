# ---- build: biên dịch TypeScript + bundle bằng Vite
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
# Có lockfile thì cài đúng phiên bản đã khóa; lần đầu chưa có thì npm install (nhớ commit package-lock.json).
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY . .
RUN npm run build

# ---- runtime: nginx phục vụ file tĩnh và proxy /v1 về backend
FROM nginx:1.27-alpine
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1/healthz >/dev/null || exit 1
