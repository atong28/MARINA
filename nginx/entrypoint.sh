#!/bin/sh
set -eu

if [ "${EXPOSE_API_VIA_NGINX:-false}" = "true" ]; then
  cp /etc/nginx/conf.d/marina-proxy-api-on.conf /etc/nginx/conf.d/marina-proxy.conf
else
  cp /etc/nginx/conf.d/marina-proxy-api-off.conf /etc/nginx/conf.d/marina-proxy.conf
fi

exec nginx -g "daemon off;"
