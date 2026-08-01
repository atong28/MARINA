# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .

# VITE_API_BASE defaults to /api so nginx can proxy it to the backend service.
# Pass a different value at build time if you need a direct URL:
#   docker build --build-arg VITE_API_BASE=https://api.example.com/api ...
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE

# Show the API Docs header link only when the edge proxy actually routes
# /docs (EXPOSE_API_VIA_NGINX=true); otherwise the link leads nowhere.
ARG VITE_SHOW_API_DOCS=false
ENV VITE_SHOW_API_DOCS=$VITE_SHOW_API_DOCS

RUN npm run build


# ── Stage 2: runtime (nginx) ──────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove the default nginx config
RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/marina.conf

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
