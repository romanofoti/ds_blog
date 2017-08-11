---
layout: post
title: Serving multiple Jekyll-GitHub sites on a custom domain
tags: jekyll github web
---

Here I share my experience of hosting two Jekyll-powered websites on GitHub (using their GitHub Pages service) and hosting serving them from one custom domain. While there are a plethora of tutorials and posts out there that show how to do this as well, everyone's needs are personal and utlimately different and, if you are making it thus far as reading this post in your search, perhaps you have not found the right one for you yet.

I hope that the information I cover here will be applicable to your usecase and will help you save some time and energy in building and hosting your website(s).


## Who can benefit from this guide

I am writing this guide for whomever wants to host multiple, independent, websites (e.g. 1 personal website and 1 blog) and have both of them powered by Jekyll, served/hosted on GitHub Pages, and accessible via custom domain.

Having the above structure has the following advantages:

- Each website can be developed independently, with specific layout and configurations.
- Using Jekyll* plus a text editor is enough to get you going, and you can test-serve each website locally during development through the jekyll local serve option.
- You can use subdomains of your personal domain to host the additional websites without worrying about conflicts.

*Note: this tutorial works even if you do not want to use any Jekyll development platform, or you want to use it only for one of your websites.

## What you need to do first

Here are some initial set-up requirements and steps:

 - Have a GitHub account
 - Have a custom domain
 - Install Jekyll (optional)
 - Create a local repository for each website you want to serve

## Setups

To have your system working, you will need to appropriately configure:

 - Your GitHub
 - Your custom domain
 - Your repositories

Note: To make things easy, I will be outlining a setup specifically tailored to the structure: 1 personal website + 1 blog (or project) website. You may be able to extrapolate and transfer most of the information I provide here to fit your specific usecase. 

### GitHub setup
 
GitHub requirements for hosting websites have slightly changed overtime, so perhaps a few weeks or months from now some of the steps below will not be fully applicable to your case. Therefore, please make sure that you adapt the information I am giving you to their most current guidelines.

As of January 2017, GitHub lets you host 1 "user" website and unlimited "project" websites through their GitHub Pages engine. Your user website will be hosted automatically from the master branch, providede that you name your repository in the right way, while - unlike until the recent the past, you can decide to build your project websites either from your master branch or from a branch specifically named "gh-pages".

For the setup I specified earlier, here are the steps to configure your GitHub as of January 2017:

 - For your primary personal website
   - Name your primary (personal website) repository as *<<your_github_username>>.github.io*.
   - Go to your repository settings and write the address of your custom domain in the custom domain field, for example *www.<<your_custom_domain_name>>.com*.
 - For your blog website
   - Name your blog repository the way you like.
   - Go to your blog repository settings and write the address of your custom subdomain in the custom domain field, for example *blog.<<your_custom_domain_name>>.com*.
   - In your blog repository settings, you can choose whether you want to build your website from the master branch or from a dedicated gh-pages branch. Now that you have this option, I would recommend you to build it from your master branch.

### Domain (DNS) setup
 
The DNS configuration steps may differ somehow among providers, at least in their "execution". However, once you log on your custom domain management console, you need to:

 - Create two A records, from your primary domain, pointing to the following IP addresses: 192.30.252.153, 192.30.252.154* 
 - Create one *Emphasize* _www_ subdomain, with an associated CNAME pointing to *<<your_github_username>>.github.io*
 - Create one *Emphasize* _blog_ subdomain, with an associated CNAME **Strong** _also_ pointing to *<<your_github_username>>.github.io*

*Some DNS providers will not let you create multiple A records, but only change their primary one. In this case, just pick one of those two and you should be fine anyway.

### Repositories setup
 
Now, you need to make sure that each of your local repositories - which you are eventually going to push to GitHub - contain one file, which you must name *CNAME*, file with the following content:

 - *www.<<your_custom_domain_name>>.com* for your personal website repository
 - *blog.<<your_custom_domain_name>>.com* for your blog repository

## Conclusion

Well, that is it, really. In retrospect, building multiple websites and hosting them on GitHub is not difficult at all, but I understand that some steps may be confusing (at least, they were to me at the beginning), so I hope that this offers some help to those of you whose structure and objectives in building their websites are similar to mine.

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>