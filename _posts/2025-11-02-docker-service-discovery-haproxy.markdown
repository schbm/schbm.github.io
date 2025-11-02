---
layout: single
title:  "Docker Service Discovery with HAProxy"
date:   2025-11-2 12:00:00 +0100
show_date: true
categories: devops guide
tags: devops docker haproxy service-discovery dns
toc: True
---

Docker (compose) has the built-in funcionality of using DNS loadbalancing with round robin on custom networks.
We can use this, to do simple service discovery when using loadbalancers like HAProxy.
The documentation for this is sadly really bad.

The documentation can be summarized with the following snippets:
- "Compose sets up a single network for your app. (...) Each container can now look up the service name 'web' or 'db' and get back the appropriate container's IP address." [^1]
- "By default Compose sets up a single network … Each container for a service joins the default network (...) and is both reachable by other containers on that network, and discoverable by the service's name." [^2]

If you've read these descriptions, you might still feel unsure about how it actually works under the hood.
My goal in this post is to show you exactly that.

# Setup
Suppose we have a (horizontally) scalable service which we want to load balance:
```yaml
services:

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    networks:
      - test_network

  haproxy:
    image: haproxy:latest
    ports:
      - "443:443"
      - "9000:9000"
    networks:
      - test_network
```

There are two quick ways to do this:
1. Use the `--scale` option to adjust the number of instances.
2. Define the number of replicas directly within the service configuration.

## Option 1
The [manual of the `up` command](https://docs.docker.com/reference/cli/docker/compose/up/) defines a neat argument
with which we can spin up multiple service replicas:
```console
--scale		Scale SERVICE to NUM instances. Overrides the scale setting in the Compose file if present.
```
Unfortunately, it doesn't explain in detail how this actually works, leaving us unsure how to adjust our HAProxy configuration accordingly.

## Option 2
According to the Compose docs, you can achieve the same scaling by setting the number of replicas in the service manifest.
```yaml
  backend:
    deploy:
      replicas: 2
```
However, the same issue remains: the documentation doesn't explain how to integrate this with our load balancer, so it's unclear how to apply it in HAProxy.

# What happens
Docker comes with a default nameserver, thats reachable at `127.0.0.11:53`
Each container automatically has an entry for this nameserver in its `/etc/resolv.conf` file.
That's the reason why you can find the service using either the `<servicename>` or `task_<servicename>_n`, with the latter returning the address of a specific replica.

We can query these entries via within a container on the same network:
```console
/app$ nslookup backend
Server:     127.0.0.11
Address:    127.0.0.11:53

Non-authoritative answer:
Name:       backend
Address:    172.18.0.6
Name:       backend
Address:    172.18.0.5
```

Or even quickly spin up a temporary container containing dig:
```console
marcel@local:~/project$ docker run --rm --network test_network tutum/dnsutils dig backend
;; QUESTION SECTION:
;backend.                                IN      A

;; ANSWER SECTION:
backend.                 600     IN      A       172.28.0.2
backend.                 600     IN      A       172.28.0.3
backend.                 600     IN      A       172.28.0.4
```

# Set HAProxy Configuration
So how can we now write our config file to accomodate this?
We will need to entities: [`resolvers`](https://www.haproxy.com/documentation/haproxy-configuration-manual/latest/#5.3.2) and 
[`server-template`](https://www.haproxy.com/documentation/haproxy-configuration-manual/latest/#4-server-template).

We set a resolver that represents our docker dns service. In its configuration we can set
optional definitions like: when the server state changes from `UP` to `down` or
the timeout duration and retry count.

After we have created such a resolver we can use the 'server-template'
keywork in our service definition with the following signature:
```
  server-template <prefix> <num | range> <fqdn>[:<port>] [params*]
```

For our example we can add:
```
  server-template backend- 2 backend:80 check resolvers docker resolve-prefer ipv4
  # equal to
  server backend-1 backend:80 check
  server backend-2 backend:80 check
``` 

Our final configuration excerpt:
```
resolvers docker
    nameserver dns 127.0.0.11:53
    resolve_retries       3
    timeout resolve       1s
    timeout retry         1s
    hold valid            10s

backend http_backend
    mode http
    balance roundrobin
    option httpchk GET /
    server-template backend- 2 backend:80 check resolvers docker resolve-prefer ipv4
```

If you paid attention it was easy to spot a problem: We need to set a concrete number of service instances.
This can be solved pretty easily in different ways: 
- Pick a really big upper-bound and let HAProxy figure out the available replicas.
- Update the HAProxy config on replica changes and restart the service.
- Use a different LB like Traefik.

[^1]: [Networking in Compose](https://docs.docker.com/compose/how-tos/networking)
[^2]: [Define and manage networks in Docker Compose](https://docs.docker.com/reference/compose-file/networks/)