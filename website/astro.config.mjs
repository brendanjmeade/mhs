// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// GitHub Pages project site: https://<user>.github.io/<repo>/
// The workflow passes SITE and BASE; the defaults are for `npm run dev`.
export default defineConfig({
  site: process.env.SITE ?? "https://brendanjmeade.github.io",
  base: process.env.BASE ?? "/mhs",
  trailingSlash: "ignore",
  compressHTML: false,
  integrations: [mdx()],
  markdown: {
    processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
  },
});
