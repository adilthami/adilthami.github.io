import type { SvgComponent } from "astro/types"
import Email from "@/assets/icons/email.svg"
import GitHub from "@/assets/icons/github.svg"
import RSS from "@/assets/icons/rss.svg"
import Twitter from "@/assets/icons/twitter.svg"
import LinkedIn from "@/assets/icons/linkedin.svg"

export const SITE = {
  title: "Adil Thami",
  description: "A personal portfolio website showcasing my projects and blog posts.",
  locale: "en-US",
  dir: "ltr",
  defaultPageImage: "/static/opengraph-image.png",
  defaultPostImage: "/static/1200x630.png",
} as const

export const NAVIGATION = [
  { href: "/blog", label: "Blog" },
  { href: "/projects", label: "Projects" },
  { href: "/about", label: "About me" },
]

export const SOCIALS: { href: string; label: string; icon: SvgComponent }[] = [
  { href: "https://github.com/adilthami", label: "GitHub", icon: GitHub },
  { href: "https://linkedin.com/in/adil-thami", label: "LinkedIn", icon: LinkedIn },
  { href: "mailto:thami.adil.pro@gmail.com", label: "Email", icon: Email },
  // { href: "/rss.xml", label: "RSS", icon: RSS },
]
