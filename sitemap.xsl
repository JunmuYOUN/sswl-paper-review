<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:sitemap="http://www.sitemaps.org/schemas/sitemap/0.9">
<xsl:output method="html" encoding="UTF-8" indent="yes"/>
<xsl:template match="/">
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sitemap — SSWL Paper Review</title>
  <style>
    :root { --deep-space:#0B1426; --sun-gold:#E8A838; --star-white:#F4F1EC; --bg:#FDFBF7; --card:#FFF; --border:#E8E4DC; --text:#1A1A2E; --sub:#4A4A6A; --light:#8888A8; }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:Arial,sans-serif; background:var(--bg); color:var(--text); line-height:1.7; }
    .header { background:linear-gradient(135deg,var(--deep-space),#1A2744,#2A3F6E); padding:40px 24px 32px; text-align:center; }
    .header h1 { color:var(--star-white); font-size:1.6rem; margin-bottom:4px; }
    .header p { color:rgba(244,241,236,0.7); font-size:0.9rem; }
    .container { max-width:900px; margin:0 auto; padding:24px; }
    .count { font-size:0.85rem; color:var(--light); margin-bottom:16px; }
    table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    th { background:var(--deep-space); color:rgba(244,241,236,0.85); font-size:0.78rem; font-weight:600; text-align:left; padding:10px 14px; text-transform:uppercase; letter-spacing:0.05em; }
    td { padding:8px 14px; border-top:1px solid var(--border); font-size:0.85rem; }
    tr:hover td { background:rgba(232,168,56,0.04); }
    a { color:#D4722A; text-decoration:none; }
    a:hover { text-decoration:underline; }
    .priority { display:inline-block; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.75rem; }
    .p-high { background:#E8F5E9; color:#2E7D32; }
    .p-mid { background:#E3F2FD; color:#1565C0; }
    .p-low { background:#FFF8E1; color:#F57F17; }
    footer { text-align:center; padding:32px 24px; font-size:0.78rem; color:var(--light); }
  </style>
</head>
<body>
  <div class="header">
    <h1>Sitemap</h1>
    <p>SSWL Paper Review — <xsl:value-of select="count(sitemap:urlset/sitemap:url)"/> pages</p>
  </div>
  <div class="container">
    <p class="count"><xsl:value-of select="count(sitemap:urlset/sitemap:url)"/> URLs indexed</p>
    <table>
      <thead>
        <tr>
          <th>URL</th>
          <th>Frequency</th>
          <th>Priority</th>
        </tr>
      </thead>
      <tbody>
        <xsl:for-each select="sitemap:urlset/sitemap:url">
          <tr>
            <td><a href="{sitemap:loc}"><xsl:value-of select="sitemap:loc"/></a></td>
            <td><xsl:value-of select="sitemap:changefreq"/></td>
            <td>
              <xsl:choose>
                <xsl:when test="sitemap:priority >= 0.9"><span class="priority p-high"><xsl:value-of select="sitemap:priority"/></span></xsl:when>
                <xsl:when test="sitemap:priority >= 0.7"><span class="priority p-mid"><xsl:value-of select="sitemap:priority"/></span></xsl:when>
                <xsl:otherwise><span class="priority p-low"><xsl:value-of select="sitemap:priority"/></span></xsl:otherwise>
              </xsl:choose>
            </td>
          </tr>
        </xsl:for-each>
      </tbody>
    </table>
  </div>
  <footer>SSWL Paper Review · Kyung Hee University</footer>
</body>
</html>
</xsl:template>
</xsl:stylesheet>
