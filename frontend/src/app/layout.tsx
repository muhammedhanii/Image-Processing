import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIU Image Toolkit",
  description: "Image Processing and Cryptography Toolkit based on CSE281 Lectures",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
