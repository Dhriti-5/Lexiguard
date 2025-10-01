import React, { useState } from "react";
import { draftNegotiation } from "../utils";

export default function NegotiationAssistant({ riskyClause }) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);

  const handleDraft = async () => {
    setLoading(true);
    const result = await draftNegotiation(riskyClause);
    setEmail(result.negotiation_email);
    setLoading(false);
  };

  return (
    <div className="p-4 border rounded-xl shadow mt-4 bg-gray-800 text-gray-100">
      <h2 className="text-xl font-bold mb-2">📧 Negotiation Assistant</h2>
      <button
        onClick={handleDraft}
        disabled={loading}
        className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-500"
      >
        {loading ? "Drafting..." : "Draft Negotiation Email"}
      </button>

      {email && (
        <div className="mt-4 p-3 border rounded bg-gray-700 text-gray-100">
          <h3 className="font-semibold">Suggested Email:</h3>
          <p className="whitespace-pre-line">{email}</p>
        </div>
      )}
    </div>
  );
}