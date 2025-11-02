import { NextResponse } from "next/server";
import { auth } from "@clerk/nextjs/server";
import connectDB from "@/lib/mongodb";
import GameResult from "@/models/GameResult";

// GET /api/games/results?gameId=first
export async function GET(request) {
  try {
    const { userId, isAuthenticated } = await auth(request);
    if (!isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    await connectDB();

    const { searchParams } = new URL(request.url);
    const gameId = searchParams.get("gameId");

    const query = { userId };
    if (gameId) query.gameId = gameId;

    const results = await GameResult.find(query).sort({ createdAt: -1 }).lean();
    return NextResponse.json(results);
  } catch (e) {
    console.error("GET /api/games/results error:", e);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// POST /api/games/results
// body: { gameId: string, score?: number, metrics?: object }
export async function POST(request) {
  try {
    const { userId, isAuthenticated } = await auth(request);
    if (!isAuthenticated || !userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    await connectDB();
    const body = await request.json();
    const { gameId, score = 0, metrics = {} } = body || {};

    if (!gameId) {
      return NextResponse.json(
        { error: "gameId is required" },
        { status: 400 }
      );
    }

    const created = await GameResult.create({ userId, gameId, score, metrics });
    return NextResponse.json(created, { status: 201 });
  } catch (e) {
    console.error("POST /api/games/results error:", e);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
