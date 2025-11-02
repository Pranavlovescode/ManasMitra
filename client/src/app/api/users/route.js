import { auth, currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import connectDB from "../../../lib/mongodb";
import User from "../../../models/User";

// GET /api/users - Get current user profile
export async function GET() {
  try {
    const { userId } = auth();

    if (!userId) {
      console.warn("Unauthorized access attempt to /api/users");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    try {
      await connectDB();
    } catch (dbError) {
      console.error("Database connection failed in users GET:", dbError);
      return NextResponse.json(
        {
          error: "Database connection failed. Please try again later.",
        },
        { status: 503 }
      );
    }

    let user = await User.findOne({ clerkId: userId });

    // If user doesn't exist in MongoDB, create them from Clerk data
    if (!user) {
      const clerkUser = await currentUser();

      if (!clerkUser) {
        return NextResponse.json(
          { error: "User not found in Clerk" },
          { status: 404 }
        );
      }

      // Create user in MongoDB with Clerk data
      user = new User({
        clerkId: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        firstName: clerkUser.firstName,
        lastName: clerkUser.lastName,
        role: clerkUser.unsafeMetadata?.role || "patient",
        profileImage: clerkUser.imageUrl,
        isActive: true,
        lastLogin: new Date(),
      });

      await user.save();
    }

    return NextResponse.json(user);
  } catch (error) {
    console.error("Error fetching user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// PUT /api/users - Update current user profile
export async function PUT(req) {
  try {
    const { userId } = auth();

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();

    await connectDB();

    let user = await User.findOne({ clerkId: userId });

    // If user doesn't exist, create them first
    if (!user) {
      const clerkUser = await currentUser();

      if (!clerkUser) {
        return NextResponse.json(
          { error: "User not found in Clerk" },
          { status: 404 }
        );
      }

      user = new User({
        clerkId: clerkUser.id,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        firstName: clerkUser.firstName,
        lastName: clerkUser.lastName,
        role: clerkUser.unsafeMetadata?.role || "patient",
        profileImage: clerkUser.imageUrl,
        isActive: true,
      });
    }

    // Update user with the provided data
    Object.assign(user, body);
    user.lastLogin = new Date();

    await user.save();

    return NextResponse.json(user);
  } catch (error) {
    console.error("Error updating user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// DELETE /api/users - Soft delete current user
export async function DELETE() {
  try {
    const { userId } = auth();

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    await connectDB();

    const user = await User.findOneAndUpdate(
      { clerkId: userId },
      { isActive: false },
      { new: true }
    );

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    return NextResponse.json({ message: "User deactivated successfully" });
  } catch (error) {
    console.error("Error deactivating user:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
