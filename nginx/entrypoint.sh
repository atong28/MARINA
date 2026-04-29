#!/bin/sh
set -eu

# Ensure only one active server config is loaded from conf.d.
rm -f /etc/nginx/conf.d/marina-proxy.conf

if [ "${EXPOSE_API_VIA_NGINX:-false}" = "true" ]; then
  cp /etc/nginx/marina/marina-proxy-api-on.conf /etc/nginx/conf.d/marina-proxy.conf
else
  cp /etc/nginx/marina/marina-proxy-api-off.conf /etc/nginx/conf.d/marina-proxy.conf
fi

exec nginx -g "daemon off;"
