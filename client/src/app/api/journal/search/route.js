import {  NextRequest, NextResponse } from "next/server"
import { db } from "@/lib/db"
import { verifyToken } from "@/lib/auth"

export async function GET(request) {
  try {
    const authHeader = request.headers.get("authorization")
    const token = authHeader?.replace("Bearer ", "")

    if (!token || !verifyToken(token)) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const userId = request.nextUrl.searchParams.get("userId")
    const query = request.nextUrl.searchParams.get("q")?.toLowerCase() || ""

    const journals = db.journals
      .filter(
        (j) =>
          j.userId === userId && (j.title.toLowerCase().includes(query) || j.content.toLowerCase().includes(query)),
      )
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())

    return NextResponse.json(journals)
  } catch (error) {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
