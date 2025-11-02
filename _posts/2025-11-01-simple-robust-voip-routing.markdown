---
layout: single
title:  "Simple and Robust VoIP Routing (to Teams)"
date:   2025-11-1 13:00:00 +0100
show_date: true
categories: system-engineering voip guide
tags: voip sbc teams direct-routing system-engineering
toc: True
---

VoIP really is a pain! Most who dealt with different PBXs, SBCs, and providers knows the only consistent thing about them is how inconsistently they work.
And do not get me started on debugging.
With the recent [demise of Skype for Business](https://support.microsoft.com/en-us/skype/skype-is-retiring-in-may-2025-what-you-need-to-know-2a7d2501-427f-485e-8be0-2068a9f90472),
you may have considered adopting Teams as a phone system. But I guess many companies have their own often complex voice ecosystems, where many problems can arise.
- How do we integrate Teams into our existing on-premises infrastructure?
- How can we ensure that, in case of an emergency, calls are connected to an appropriate receiver?
- How can we manage our registered phone numbers?

There are obviously many more questions one must ask in this endeavour.
But in this post, I want to present a simple and *easy to implement* enterprise architecture and focus on the main question of call routing.

A few prerequisites:
- An existing on premise voice infrastructure. In this guide i will use well known sbc devices from [Audiocodes](https://www.audiocodes.com/de/solutions-products/products/session-border-controllers-sbcs) to be more specific the mediant product line.
A list of supported devices can be found [here](https://learn.microsoft.com/de-de/microsoftteams/direct-routing-border-controllers)
- A SBC is already connected to [Teams Direct Routing](https://learn.microsoft.com/de-de/microsoftteams/direct-routing-plan)
- An existing [directory service](https://en.wikipedia.org/wiki/List_of_LDAP_software) exists with the support of [LDAP](https://en.wikipedia.org/wiki/Lightweight_Directory_Access_Protocol)

<figure style="max-width:100%;">
  <img src="/assets/images/2025-01-11-simple-robust-voip-routing/overview.drawio.svg"
       alt="Target Architecture"
       style="width:100%;">
  <figcaption style="text-align:center;">Target Architecture</figcaption>
</figure>

1. The SBC (or SBC cluster) functions as the heart of operations. It connects all peers and determines how to handle our VoIP traffic.
2. With the help of directory services, we can make informed routing decisions. However, there is certainly no single correct way of doing this.
For example, within the Mediant product line, one could handle routing using so-called dial plans, where specific source or target numbers are matched and tagged for routing purposes.
Another viable approach is to base routing directly on specific LDAP attributes, which can be very beneficial due to the ability to store all relevant information within the directory.
This also comes with the advantage that some models, such as the Mediant, can cache LDAP results, making the system more robust against connection failures.
These two approaches can easily handle thousands of customers, but in the end, even manually creating individual rules may suffice if the data does not change often.
3. The architecture, if used correctly, is agnostic to the number and specific types of external providers.
4. The same holds true for the internal network(s).
5. And for Teams, which essentially functions as just another trunk.

# Action Plan
For the sake of simplicity, I'll use dial plans in this guide.
The entries can be imported or exported programmatically.
If you only have a standard Mediant, you may need to fetch the entire configuration file and parse the dial plan entries separately, since the latest API version still does not support a simple `GET`.
Uploads, however, are supported via `PUT`.
I'll leave it to the reader to choose a method appropriate for their system.
The general routing process is shown in the figure below.

It works like this:
1. The Mediant classifies the number.
The specific classification rules or logic I won't show here.
However, if you, for example, have an IP Group with a defined Proxy Set, you can enable automatic classification for this group.
The SBC then checks the SIP interface that received the call and the source address, matching it to the corresponding IP Group. If you have a group for individual registered VoIP phones, you must add an additional classification rule.
These details are very specific to the device you are using.
2. The device then matches defined rules to the received source and target addresses and adds an internal tag.
3. Next, it queries the LDAP server. The Mediant product line handles queries in different ways. You can define a Call Setup Rule, for example, for a specific IP Group. This rule runs after the dial plan lookup and before routing.
The rules can include script-like logic, allowing you to perform specific LDAP attribute lookups.
4. Once the query is complete and the correct routing rule and tag are found, the call can be routed to its destination.

<figure style="max-width:100%;">
  <a href="/assets/images/2025-01-11-simple-robust-voip-routing/routing.drawio.svg" target="_blank" rel="noopener">
    <img src="/assets/images/2025-01-11-simple-robust-voip-routing/routing.drawio.svg"
         alt="The routing process (BPMN 2.0)"
         style="width:100%; cursor: zoom-in;">
  </a>
  <figcaption style="text-align:center;">The call routing process (BPMN 2.0) (click to view full size)</figcaption>
</figure>

## Handling Distasters

Below, I expand on the *route call task* to handle cases in which Teams is down.
If call processing times out or an error response is received, we query the LDAP server again to retrieve an attribute of our choice.
This attribute can store additional information on how to handle the call, for example, an alternative number.
If such a number is found, we can simply restart the routing process from the beginning.
This structure requires that your dial plan rules are correct and contain only values that can actually be routed.
Otherwise, this could lead to unexpected behaviour, where the system queries for alternative routing information for calls that should not be routed in the first place.

<figure style="max-width:100%;">
  <a href="/assets/images/2025-01-11-simple-robust-voip-routing/disaster.drawio.svg" target="_blank" rel="noopener">
    <img src="/assets/images/2025-01-11-simple-robust-voip-routing/disaster.drawio.svg"
         alt="The expanded call routing process (BPMN 2.0)"
         style="width:100%; cursor: zoom-in;">
  </a>
  <figcaption style="text-align:center;">The expanded call routing process (BPMN 2.0) (click to view full size)</figcaption>
</figure>

# Implementation on Audiocodes SBC
To implement the afforementioned processes all we need in the SBC are a few new core entities.

<figure style="max-width:100%;">
  <a href="/assets/images/2025-01-11-simple-robust-voip-routing/mediant-config.drawio.svg" target="_blank" rel="noopener">
    <img src="/assets/images/2025-01-11-simple-robust-voip-routing/mediant-config.drawio.svg"
         alt="Audiocodes Mediant Config"
         style="width:100%; cursor: zoom-in;">
  </a>
  <figcaption style="text-align:center;">Required Audiocodes Mediant Core Entities</figcaption>
</figure>

In this guide i will not further explain the meaning of those entities. So if you have the use case follow the [manual](https://www.audiocodes.com/media/5l4nye5a/mediant-software-ve-ce-se-sbc-users-manual-ver-76.pdf).

## SIP-Interfaces
Create two SIP-Interfaces one for the communication to the provider and one to Teams.

Within the interface set the following:
- TCP/UDP settings
- Network inteface that faces to the external devices

## Proxy-Sets
Next, create two proxy sets that define the addresses of the provider and Teams devices.

Set the following:
- Proxy Addresses (DNS or IP)
- Configure the authentication. My favourite would be certificate based auth. The only big deficit of this is the rotation, which has to be done regularly.
- Configure keep-alive
These settings highly depend on your communication partner.

## IP-Group
Now, representing the SIP devices, create two IP groups.

Use the following:
- Use the Proxy-Sets created before
- Reuse or create a new mediarealm
- Enable *classify by proxy set*. When you enable this you do not need to create additional classification rules.
- Create a new dialplan and use it for both IP-Groups
- Create or reuse alternative routing reasons e.g for all 4XX and 5XX

## Top it off with IP2IP Rules

So to finish this off, all you have to do is to create a single rule,
that that matches on calls tagged with e.g *teams* to the IP-Group.
For the disaster case, create an alternative route right below and 
handle your logic by setting (pre) call setup rules, in which you e.g set it
to another target number or tag! 

![Rule](/assets/images/2025-01-11-simple-robust-voip-routing/rule.png)

![Rule](/assets/images/2025-01-11-simple-robust-voip-routing/rule1.png)

# Conclusion
With this setup, you get a simple and extensible way to connect Teams (or any other target) to your existing VoIP environment.
Using dial plans, LDAP lookups, and a bit of routing logic, you can keep calls flowing smoothly, even if something goes down.
It's an easy approach that still leaves plenty of room to grow and adjust as your system changes.