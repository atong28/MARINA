FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf

RUN mkdir -p /etc/nginx/marina
COPY nginx.conf /etc/nginx/marina/marina-proxy-api-on.conf
COPY nginx.no-api.conf /etc/nginx/marina/marina-proxy-api-off.conf
COPY security-headers.conf security-headers-docs.conf security-headers-worker.conf /etc/nginx/marina/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80

CMD ["/entrypoint.sh"]
