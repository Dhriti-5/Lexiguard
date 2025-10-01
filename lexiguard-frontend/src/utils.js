// src/utils.js

const BASE_URL = process.env.REACT_APP_BACKEND_URL; // or your deployed Cloud Run URL

export async function analyzeExtended(text) {
  const res = await fetch(`${BASE_URL}/analyze-extended`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function draftNegotiation(clause) {
  const res = await fetch(`${BASE_URL}/draft-negotiation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clause }),
  });
  return res.json();
}

export function createPageUrl(pageName) {
  return `/${pageName.toLowerCase()}`;
}
