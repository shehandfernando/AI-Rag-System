import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG System",
  description: "A grounded PDF assistant with validated citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
