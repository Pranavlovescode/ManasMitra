import * as React from "react"
import { cn } from "@/lib/utils"

const Collapsible = React.forwardRef(({ className, ...props }, ref) => (
  <div
    className={cn("", className)}
    ref={ref}
    {...props}
  />
))
Collapsible.displayName = "Collapsible"

const CollapsibleTrigger = React.forwardRef(({ className, ...props }, ref) => (
  <button
    className={cn("", className)}
    ref={ref}
    {...props}
  />
))
CollapsibleTrigger.displayName = "CollapsibleTrigger"

const CollapsibleContent = React.forwardRef(({ className, ...props }, ref) => (
  <div
    className={cn("overflow-hidden", className)}
    ref={ref}
    {...props}
  />
))
CollapsibleContent.displayName = "CollapsibleContent"

export {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
}