"use client";

import { cn } from "@/lib/utils";
import { useState } from "react";

export default function DashboardSidebar({ items, activeTab, onTabChange, userInfo }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 h-full bg-white/90 backdrop-blur-md border-r border-gray-200 shadow-lg transition-all duration-300 z-40",
          collapsed ? "w-20" : "w-64"
        )}
      >
        {/* Logo/Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {!collapsed && (
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white text-xl font-bold">
                    {userInfo?.initial || "M"}
                  </span>
                </div>
                <div>
                  <h2 className="font-semibold text-gray-800 text-sm">
                    {userInfo?.name || "User"}
                  </h2>
                  <p className="text-xs text-gray-500">{userInfo?.role || "Dashboard"}</p>
                </div>
              </div>
            )}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className={cn(
                "p-2 hover:bg-gray-100 rounded-lg transition-colors",
                collapsed && "mx-auto"
              )}
            >
              <svg
                className={cn(
                  "w-5 h-5 text-gray-600 transition-transform",
                  collapsed && "rotate-180"
                )}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M11 19l-7-7 7-7m8 14l-7-7 7-7"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-4 space-y-2 overflow-y-auto h-[calc(100vh-120px)]">
          {items.map((item) => (
            <button
              key={item.value}
              onClick={() => onTabChange(item.value)}
              className={cn(
                "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
                activeTab === item.value
                  ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md"
                  : "text-gray-700 hover:bg-gray-100"
              )}
            >
              <span className="text-xl flex-shrink-0">{item.icon}</span>
              {!collapsed && (
                <span className="font-medium text-sm">{item.label}</span>
              )}
              {!collapsed && item.badge && (
                <span
                  className={cn(
                    "ml-auto px-2 py-0.5 text-xs rounded-full",
                    activeTab === item.value
                      ? "bg-white/20 text-white"
                      : "bg-red-100 text-red-600"
                  )}
                >
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </nav>
      </aside>

      {/* Spacer to prevent content from going under sidebar */}
      <div className={cn("transition-all duration-300", collapsed ? "w-20" : "w-64")} />
    </>
  );
}
