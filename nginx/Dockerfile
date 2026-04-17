FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/marina-proxy.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
